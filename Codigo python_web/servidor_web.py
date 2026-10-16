"""
============================================================
 ESP32-CAM + YOLO  |  DETECCIÓN EN TIEMPO REAL EN UNA PÁGINA WEB
 Taller ULIMA 2026 - Laboratorio de Internet de las Cosas
============================================================

¿Qué hace este programa?

  1. Se conecta al video en vivo de la ESP32-CAM  (http://IP:81/stream)
  2. Analiza cada fotograma con el modelo YOLO11n
  3. Levanta una PÁGINA WEB en la misma PC:  http://localhost:5000
  4. Abre el navegador automáticamente para ver la detección en vivo.

        ESP32-CAM  --Wi-Fi-->  PC (Python + YOLO + página web)

Instalación (una sola vez):
    pip install ultralytics flask requests

Ejecución:
    python servidor_web.py

(Opcional) Para que otros dispositivos de la red también vean la página,
cambia COMPARTIR_EN_RED = True e instala:  pip install qrcode
"""

import io
import os
import socket
import webbrowser
import threading
import time
import unicodedata
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN  (lo único que normalmente hay que cambiar)
# ============================================================

ESP32_IP = "192.168.51.170"           # <-- CAMBIAR por la IP de tu ESP32-CAM

# Fuente del video:
#   "esp32"  -> usa la ESP32-CAM (lo normal en el taller)
#   "webcam" -> usa la cámara de la laptop (útil para probar sin ESP32)
FUENTE = "esp32"

URL_STREAM = f"http://{ESP32_IP}:81/stream"
model = YOLO("yolo26n.pt")
MODELO_PATH = "yolo26n.pt"   # Modelo preentrenado de Ultralytics (80 objetos)
CONFIANZA_INICIAL = 0.30     # Se puede cambiar desde la página web
TAMANO_IMAGEN = 320          # 320 = más rápido | 640 = más preciso
MAX_OBJETOS = 15

PUERTO_WEB = 5000

# False -> la página solo se abre en esta PC (http://localhost:5000)  [taller]
# True  -> también pueden entrar celulares/laptops de la misma red Wi-Fi
COMPARTIR_EN_RED = False

CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA_FOTOS = os.path.join(CARPETA_BASE, "fotos")
CARPETA_RESULTADOS = os.path.join(CARPETA_BASE, "resultados_yolo")
os.makedirs(CARPETA_FOTOS, exist_ok=True)
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


# ============================================================
# NOMBRES DE LOS 80 OBJETOS (dataset COCO) EN ESPAÑOL
# ============================================================

NOMBRES_ES = {
    0: "persona", 1: "bicicleta", 2: "auto", 3: "motocicleta",
    4: "avión", 5: "autobús", 6: "tren", 7: "camión", 8: "barco",
    9: "semáforo", 10: "hidrante", 11: "señal de pare",
    12: "parquímetro", 13: "banca", 14: "pájaro", 15: "gato",
    16: "perro", 17: "caballo", 18: "oveja", 19: "vaca",
    20: "elefante", 21: "oso", 22: "cebra", 23: "jirafa",
    24: "mochila", 25: "paraguas", 26: "bolso", 27: "corbata",
    28: "maleta", 29: "frisbee", 30: "esquís", 31: "snowboard",
    32: "pelota", 33: "cometa", 34: "bate de béisbol",
    35: "guante de béisbol", 36: "patineta", 37: "tabla de surf",
    38: "raqueta de tenis", 39: "botella", 40: "copa", 41: "taza",
    42: "tenedor", 43: "cuchillo", 44: "cuchara", 45: "tazón",
    46: "plátano", 47: "manzana", 48: "sándwich", 49: "naranja",
    50: "brócoli", 51: "zanahoria", 52: "hot dog", 53: "pizza",
    54: "dona", 55: "pastel", 56: "silla", 57: "sofá",
    58: "planta en maceta", 59: "cama", 60: "mesa", 61: "inodoro",
    62: "televisor", 63: "laptop", 64: "mouse", 65: "control remoto",
    66: "teclado", 67: "celular", 68: "microondas", 69: "horno",
    70: "tostadora", 71: "lavadero", 72: "refrigeradora", 73: "libro",
    74: "reloj", 75: "florero", 76: "tijeras", 77: "oso de peluche",
    78: "secadora de cabello", 79: "cepillo de dientes",
}


def sin_tildes(texto):
    """OpenCV no puede dibujar tildes ni ñ, así que las quitamos para el video."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


# Colores distintos para cada objeto (formato BGR de OpenCV)
def color_clase(clase):
    paleta = [
        (0, 140, 255), (255, 170, 0), (60, 200, 60), (200, 80, 255),
        (0, 220, 220), (255, 90, 90), (120, 255, 160), (255, 0, 160),
    ]
    return paleta[clase % len(paleta)]


# ============================================================
# ESTADO COMPARTIDO ENTRE HILOS
# ============================================================

lock = threading.Lock()

estado = {
    "conectado": False,
    "error": None,
    "frame": None,             # último fotograma original
    "numero_frame": 0,
    "jpeg_procesado": None,    # último fotograma con cajas (JPEG)
    "detecciones": [],         # lista de dicts {clase, nombre, confianza}
    "tiempo_inferencia": 0.0,  # en segundos
    "fps": 0.0,
    "confianza": CONFIANZA_INICIAL,
    "descubiertos": set(),     # clases vistas desde que inició el servidor
}


# ============================================================
# HILO 1: RECIBIR VIDEO DE LA ESP32-CAM
# ============================================================

def recibir_esp32():
    """Lee el stream MJPEG de la ESP32 y guarda siempre el último fotograma."""
    while True:
        buffer = bytearray()
        try:
            print(f"Conectando con {URL_STREAM} ...")
            with requests.get(URL_STREAM, stream=True, timeout=(5, 15)) as r:
                r.raise_for_status()
                with lock:
                    estado["conectado"] = True
                    estado["error"] = None
                print("ESP32-CAM conectada.")

                for bloque in r.iter_content(chunk_size=4096):
                    if not bloque:
                        continue
                    buffer.extend(bloque)

                    # Un JPEG empieza con FFD8 y termina con FFD9
                    while True:
                        inicio = buffer.find(b"\xff\xd8")
                        if inicio < 0:
                            if len(buffer) > 2_000_000:
                                buffer.clear()
                            break
                        fin = buffer.find(b"\xff\xd9", inicio + 2)
                        if fin < 0:
                            if inicio > 0:
                                del buffer[:inicio]
                            break

                        jpeg = bytes(buffer[inicio:fin + 2])
                        del buffer[:fin + 2]

                        frame = cv2.imdecode(
                            np.frombuffer(jpeg, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if frame is not None:
                            with lock:
                                estado["frame"] = frame
                                estado["numero_frame"] += 1

        except Exception as error:
            with lock:
                estado["conectado"] = False
                estado["error"] = str(error)
            print(f"No se pudo leer la ESP32-CAM: {error}")
            print("Reintentando en 2 segundos... (¿IP correcta? ¿misma red?)")
            time.sleep(2)


def recibir_webcam():
    """Alternativa para practicar sin ESP32: usa la cámara de la laptop."""
    camara = cv2.VideoCapture(0)
    with lock:
        estado["conectado"] = camara.isOpened()
    while True:
        ok, frame = camara.read()
        if not ok:
            with lock:
                estado["conectado"] = False
                estado["error"] = "No se pudo leer la webcam"
            time.sleep(1)
            continue
        with lock:
            estado["conectado"] = True
            estado["frame"] = frame
            estado["numero_frame"] += 1
        time.sleep(0.03)


# ============================================================
# HILO 2: YOLO ANALIZA LOS FOTOGRAMAS
# ============================================================

def dibujar(frame, detecciones):
    imagen = frame.copy()
    for d in detecciones:
        x1, y1, x2, y2 = d["caja"]
        color = color_clase(d["clase"])
        cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)

        etiqueta = f"{sin_tildes(d['nombre'])} {d['confianza']:.0%}"
        (w, h), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_txt = max(y1, h + 8)
        cv2.rectangle(imagen, (x1, y_txt - h - 8), (x1 + w + 8, y_txt), color, -1)
        cv2.putText(imagen, etiqueta, (x1 + 4, y_txt - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return imagen


def ejecutar_yolo(modelo):
    ultimo_numero = -1
    tiempos = []

    while True:
        with lock:
            frame = estado["frame"]
            numero = estado["numero_frame"]
            confianza = estado["confianza"]

        # Esperar a que llegue un fotograma nuevo
        if frame is None or numero == ultimo_numero:
            time.sleep(0.005)
            continue
        ultimo_numero = numero

        inicio = time.monotonic()
        try:
            resultado = modelo.predict(
                source=frame, conf=confianza, imgsz=TAMANO_IMAGEN,
                max_det=MAX_OBJETOS, verbose=False,
            )[0]
        except Exception as error:
            print(f"Error de YOLO: {error}")
            time.sleep(0.2)
            continue
        duracion = time.monotonic() - inicio

        detecciones = []
        for caja in resultado.boxes:
            clase = int(caja.cls[0])
            detecciones.append({
                "clase": clase,
                "nombre": NOMBRES_ES.get(clase, modelo.names[clase]),
                "nombre_en": modelo.names[clase],
                "confianza": round(float(caja.conf[0]), 3),
                "caja": [int(v) for v in caja.xyxy[0].tolist()],
            })

        procesado = dibujar(frame, detecciones)
        ok, jpeg = cv2.imencode(".jpg", procesado, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # FPS = cuántas imágenes analiza YOLO por segundo
        ahora = time.monotonic()
        tiempos = [t for t in tiempos if ahora - t < 2.0] + [ahora]

        with lock:
            if ok:
                estado["jpeg_procesado"] = jpeg.tobytes()
            estado["detecciones"] = detecciones
            estado["tiempo_inferencia"] = duracion
            estado["fps"] = len(tiempos) / 2.0
            estado["descubiertos"].update(d["clase"] for d in detecciones)


# ============================================================
# SERVIDOR WEB (Flask)
# ============================================================

app = Flask(__name__, template_folder=os.path.join(CARPETA_BASE, "templates"))


def ip_local():
    """Averigua la IP de esta PC en la red Wi-Fi."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no envía datos, solo elige la interfaz
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


URL_LOCAL = f"http://localhost:{PUERTO_WEB}"
URL_PAGINA = f"http://{ip_local()}:{PUERTO_WEB}" if COMPARTIR_EN_RED else URL_LOCAL


@app.route("/")
def inicio():
    return render_template(
        "index.html",
        url_pagina=URL_PAGINA,
        compartir=COMPARTIR_EN_RED,
        clases=[{"id": i, "nombre": n} for i, n in NOMBRES_ES.items()],
        confianza=CONFIANZA_INICIAL,
    )


@app.route("/imagen.jpg")
def imagen():
    """Última imagen con las cajas de YOLO (la página la pide una y otra vez)."""
    with lock:
        jpeg = estado["jpeg_procesado"]
    if jpeg is None:
        return "Todavía no hay video", 503
    return Response(jpeg, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/video")
def video():
    """Video en vivo con las cajas de YOLO (MJPEG: una imagen tras otra)."""
    def generar():
        ultimo = None
        while True:
            with lock:
                jpeg = estado["jpeg_procesado"]
            if jpeg is None or jpeg is ultimo:
                time.sleep(0.02)
                continue
            ultimo = jpeg
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")

    return Response(generar(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/estado")
def api_estado():
    with lock:
        detecciones = list(estado["detecciones"])
        datos = {
            "conectado": estado["conectado"],
            "error": estado["error"],
            "fps": round(estado["fps"], 1),
            "tiempo_ms": round(estado["tiempo_inferencia"] * 1000),
            "confianza": estado["confianza"],
            "descubiertos": sorted(estado["descubiertos"]),
        }
    conteo = Counter(d["nombre"] for d in detecciones)
    datos["detecciones"] = [
        {"nombre": d["nombre"], "nombre_en": d["nombre_en"],
         "clase": d["clase"], "confianza": d["confianza"]}
        for d in sorted(detecciones, key=lambda d: -d["confianza"])
    ]
    datos["conteo"] = conteo.most_common()
    datos["total"] = len(detecciones)
    return jsonify(datos)


@app.route("/api/confianza", methods=["POST"])
def api_confianza():
    valor = float(request.get_json(force=True).get("confianza", CONFIANZA_INICIAL))
    valor = min(max(valor, 0.05), 0.95)
    with lock:
        estado["confianza"] = valor
    return jsonify({"confianza": valor})


@app.route("/api/foto", methods=["POST"])
def api_foto():
    """Guarda la foto original y la foto con detecciones en la PC."""
    with lock:
        frame = estado["frame"]
        jpeg = estado["jpeg_procesado"]
    if frame is None or jpeg is None:
        return jsonify({"ok": False, "mensaje": "Todavía no hay video"}), 400
    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(CARPETA_FOTOS, f"foto_{fecha}.jpg"), frame)
    with open(os.path.join(CARPETA_RESULTADOS, f"resultado_{fecha}.jpg"), "wb") as f:
        f.write(jpeg)
    return jsonify({"ok": True, "mensaje": f"Foto guardada ({fecha})"})


@app.route("/api/reiniciar_reto", methods=["POST"])
def api_reiniciar():
    with lock:
        estado["descubiertos"].clear()
    return jsonify({"ok": True})


@app.route("/qr.png")
def qr():
    """Código QR con la dirección de la página (si está instalado 'qrcode')."""
    try:
        import qrcode
    except ImportError:
        return "Instala qrcode: pip install qrcode", 404
    imagen = qrcode.make(URL_PAGINA)
    salida = io.BytesIO()
    imagen.save(salida, format="PNG")
    return Response(salida.getvalue(), mimetype="image/png")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    ruta_modelo = os.path.join(CARPETA_BASE, MODELO_PATH)
    if not os.path.isfile(ruta_modelo):
        ruta_modelo = MODELO_PATH  # Ultralytics lo descarga automáticamente
    print("Cargando YOLO...")
    modelo = YOLO(ruta_modelo)

    fuente = recibir_webcam if FUENTE == "webcam" else recibir_esp32
    threading.Thread(target=fuente, daemon=True).start()
    threading.Thread(target=ejecutar_yolo, args=(modelo,), daemon=True).start()

    print("\n" + "=" * 60)
    print("  PÁGINA WEB LISTA. Se abrirá sola en el navegador.")
    print("  Si no se abre, escribe en Chrome/Edge:")
    print(f"\n      {URL_LOCAL}\n")
    if COMPARTIR_EN_RED:
        print(f"  Desde otros dispositivos de la red: {URL_PAGINA}")
    print("  Para detener el servidor: Ctrl + C")
    print("=" * 60 + "\n")

    # Abrir el navegador automáticamente después de 1,5 segundos
    threading.Timer(1.5, lambda: webbrowser.open(URL_LOCAL)).start()

    # 127.0.0.1 = solo esta PC  |  0.0.0.0 = cualquier dispositivo de la red
    host = "0.0.0.0" if COMPARTIR_EN_RED else "127.0.0.1"
    app.run(host=host, port=PUERTO_WEB, threaded=True, debug=False)


if __name__ == "__main__":
    main()
