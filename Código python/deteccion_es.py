
import os
import time
import threading
from datetime import datetime

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURACION
# ============================================================

ESP32_IP = "192.168.51.158"
URL_STREAM = f"http://{ESP32_IP}:81/stream"
MODELO_PATH = "yolo11n.pt"

CONFIANZA = 0.35
TAMANO_IMAGEN = 320

# Espera entre inferencias. El video sigue actualizandose.
INTERVALO_INFERENCIA = 0.25

MAX_OBJETOS = 10

# Ajustes para reducir la carga de CPU.
torch.set_num_threads(2)
cv2.setNumThreads(1)

CARPETA_FOTOS = "fotos"
CARPETA_RESULTADOS = "resultados_yolo"
CARPETA_VIDEOS = "grabaciones"

for carpeta in (CARPETA_FOTOS, CARPETA_RESULTADOS, CARPETA_VIDEOS):
    os.makedirs(carpeta, exist_ok=True)

NOMBRE_VENTANA = "ESP32-CAM - YOLO"


# ============================================================
# ETIQUETAS EN ESPANOL
# ============================================================

NOMBRES_ES = {
    0: "persona", 1: "bicicleta", 2: "auto",
    3: "motocicleta", 4: "avion", 5: "autobus",
    6: "tren", 7: "camion", 8: "barco",
    9: "semaforo", 10: "hidrante", 11: "senal de pare",
    12: "parquimetro", 13: "banco", 14: "pajaro",
    15: "gato", 16: "perro", 17: "caballo",
    18: "oveja", 19: "vaca", 20: "elefante",
    21: "oso", 22: "cebra", 23: "jirafa",
    24: "mochila", 25: "paraguas", 26: "bolso",
    27: "corbata", 28: "maleta", 29: "frisbee",
    30: "esquis", 31: "tabla de snowboard",
    32: "pelota deportiva", 33: "cometa",
    34: "bate de beisbol", 35: "guante de beisbol",
    36: "patineta", 37: "tabla de surf",
    38: "raqueta de tenis", 39: "botella",
    40: "copa", 41: "taza", 42: "tenedor",
    43: "cuchillo", 44: "cuchara", 45: "tazon",
    46: "platano", 47: "manzana", 48: "sandwich",
    49: "naranja", 50: "brocoli", 51: "zanahoria",
    52: "hot dog", 53: "pizza", 54: "dona",
    55: "pastel", 56: "silla", 57: "sofa",
    58: "planta", 59: "cama", 60: "mesa",
    61: "inodoro", 62: "televisor",
    63: "portatil", 64: "mouse", 65: "control remoto",
    66: "teclado", 67: "celular", 68: "microondas",
    69: "horno", 70: "tostadora", 71: "lavadero",
    72: "refrigeradora", 73: "libro", 74: "reloj",
    75: "florero", 76: "tijeras", 77: "peluche",
    78: "secadora", 79: "cepillo de dientes"
}


# ============================================================
# ESTADO COMPARTIDO
# ============================================================

lock = threading.Lock()

estado = {
    "activo": True,
    "conectado": False,
    "frame": None,
    "numero_frame": 0,
    "detecciones": [],
    "cantidad": 0,
    "deteccion_activa": True,
    "error": None,
    "tiempo_inferencia": 0.0
}

ultimo_original = None
ultimo_procesado = None

grabador = None
grabacion_activa = False
ruta_grabacion = ""

botones = []


# ============================================================
# GUARDAR IMAGENES
# ============================================================

def guardar_imagen(imagen, carpeta, prefijo):
    if imagen is None:
        print("Todavia no hay una imagen disponible.")
        return

    fecha = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ruta = os.path.join(carpeta, f"{prefijo}_{fecha}.jpg")

    if cv2.imwrite(ruta, imagen):
        print(f"Imagen guardada: {ruta}")
    else:
        print("No se pudo guardar la imagen.")


# ============================================================
# RECEPCION DEL STREAM
# ============================================================

def recibir_video():
    while estado["activo"]:
        buffer = bytearray()

        try:
            print(f"Conectando con {URL_STREAM}...")

            with requests.get(
                URL_STREAM,
                stream=True,
                timeout=(5, 15)
            ) as respuesta:

                respuesta.raise_for_status()

                with lock:
                    estado["conectado"] = True
                    estado["error"] = None

                print("Camara conectada.")

                for bloque in respuesta.iter_content(chunk_size=1024):
                    if not estado["activo"]:
                        break

                    if not bloque:
                        continue

                    buffer.extend(bloque)

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
                            cv2.IMREAD_COLOR
                        )

                        if frame is not None:
                            # Conservar siempre el ultimo fotograma.
                            with lock:
                                estado["frame"] = frame
                                estado["numero_frame"] += 1

        except Exception as error:
            with lock:
                estado["conectado"] = False
                estado["error"] = str(error)

            if estado["activo"]:
                print(f"Error de conexion: {error}")
                time.sleep(2)

        finally:
            with lock:
                estado["conectado"] = False


# ============================================================
# YOLO: HILO INDEPENDIENTE
# ============================================================

def ejecutar_deteccion(modelo):
    ultimo_numero = -1

    while estado["activo"]:

        with lock:
            activa = estado["deteccion_activa"]
            frame = estado["frame"]
            numero = estado["numero_frame"]

            if frame is not None:
                frame = frame.copy()

        if not activa or frame is None:
            time.sleep(0.03)
            continue

        # No procesar otra vez el mismo fotograma.
        if numero == ultimo_numero:
            time.sleep(0.005)
            continue

        inicio = time.monotonic()

        try:
            resultados = modelo.predict(
                source=frame,
                conf=CONFIANZA,
                imgsz=TAMANO_IMAGEN,
                max_det=MAX_OBJETOS,
                verbose=False
            )

            resultado = resultados[0]
            detecciones = []

            if resultado.boxes is not None:
                for caja in resultado.boxes:
                    clase = int(caja.cls[0].item())
                    confianza = float(caja.conf[0].item())

                    x1, y1, x2, y2 = map(
                        int, caja.xyxy[0].tolist()
                    )

                    nombre = NOMBRES_ES.get(
                        clase,
                        str(modelo.names.get(clase, clase))
                    )

                    detecciones.append(
                        (x1, y1, x2, y2, nombre, confianza)
                    )

            duracion = time.monotonic() - inicio

            with lock:
                if estado["deteccion_activa"]:
                    estado["detecciones"] = detecciones
                    estado["cantidad"] = len(detecciones)
                    estado["tiempo_inferencia"] = duracion

            ultimo_numero = numero

        except Exception as error:
            print(f"Error de YOLO: {error}")
            time.sleep(0.2)

        restante = INTERVALO_INFERENCIA - (
            time.monotonic() - inicio
        )

        if restante > 0:
            time.sleep(restante)


# ============================================================
# DIBUJAR DETECCIONES
# ============================================================

def dibujar_detecciones(frame, detecciones):
    imagen = frame.copy()
    alto, ancho = imagen.shape[:2]

    for x1, y1, x2, y2, nombre, confianza in detecciones:
        x1 = max(0, min(x1, ancho - 1))
        x2 = max(0, min(x2, ancho - 1))
        y1 = max(0, min(y1, alto - 1))
        y2 = max(0, min(y2, alto - 1))

        cv2.rectangle(
            imagen,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        etiqueta = f"{nombre} {confianza:.0%}"

        fuente = cv2.FONT_HERSHEY_SIMPLEX
        escala = 0.48
        grosor = 1

        (ancho_texto, alto_texto), _ = cv2.getTextSize(
            etiqueta, fuente, escala, grosor
        )

        y_etiqueta = max(y1, alto_texto + 8)

        cv2.rectangle(
            imagen,
            (x1, y_etiqueta - alto_texto - 8),
            (x1 + ancho_texto + 8, y_etiqueta),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            imagen,
            etiqueta,
            (x1 + 4, y_etiqueta - 4),
            fuente,
            escala,
            (0, 0, 0),
            grosor,
            cv2.LINE_AA
        )

    return imagen


# ============================================================
# INTERFAZ SIMPLE: BOTONES
# ============================================================

def dibujar_boton(imagen, x, y, ancho, alto, texto, color):
    cv2.rectangle(
        imagen, (x, y), (x + ancho, y + alto), color, -1
    )

    cv2.rectangle(
        imagen,
        (x, y),
        (x + ancho, y + alto),
        (170, 170, 170),
        1
    )

    escala = 0.48
    grosor = 1

    tam, _ = cv2.getTextSize(
        texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor
    )

    tx = x + max(4, (ancho - tam[0]) // 2)
    ty = y + (alto + tam[1]) // 2 - 2

    cv2.putText(
        imagen,
        texto,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        (255, 255, 255),
        grosor,
        cv2.LINE_AA
    )


def crear_interfaz(video, cantidad, conectado, deteccion_activa):
    alto, ancho = video.shape[:2]

    alto_info = 32
    alto_botones = 56

    info = np.zeros((alto_info, ancho, 3), dtype=np.uint8)
    barra = np.zeros((alto_botones, ancho, 3), dtype=np.uint8)

    conexion = "CONECTADA" if conectado else "DESCONECTADA"
    deteccion = "ACTIVA" if deteccion_activa else "PAUSADA"

    texto = (
        f"Camara: {conexion} | YOLO: {deteccion} | "
        f"Objetos: {cantidad}"
    )

    cv2.putText(
        info,
        texto,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    margen = 8
    separacion = 6
    cantidad_botones = 5

    ancho_boton = (
        ancho - 2 * margen - separacion * (cantidad_botones - 1)
    ) // cantidad_botones

    nombres = [
        "Guardar foto",
        "Guardar YOLO",
        "Grabar video" if not grabacion_activa else "Detener grabacion",
        "Pausar YOLO" if deteccion_activa else "Iniciar YOLO",
        "Cerrar"
    ]

    colores = [
        (55, 110, 55),
        (65, 80, 145),
        (45, 100, 160) if not grabacion_activa else (35, 35, 190),
        (110, 85, 35),
        (40, 40, 180)
    ]

    global botones
    botones = []

    for i, nombre in enumerate(nombres):
        x = margen + i * (ancho_boton + separacion)

        dibujar_boton(
            barra, x, 7, ancho_boton, 42, nombre, colores[i]
        )

        botones.append(
            (x, 7, x + ancho_boton, 49, i)
        )

    return np.vstack((video, info, barra))


# ============================================================
# CLICS EN LOS BOTONES
# ============================================================

def manejar_clic(evento, x, y, flags, parametros):
    global grabador, grabacion_activa, ruta_grabacion

    if evento != cv2.EVENT_LBUTTONDOWN:
        return

    with lock:
        frame = estado["frame"]

        if frame is not None:
            frame = frame.copy()

        detecciones = list(estado["detecciones"])

    if frame is None:
        return

    alto_video = frame.shape[0]
    alto_info = 32

    if y < alto_video + alto_info:
        return

    y_barra = y - alto_video - alto_info

    for x1, y1, x2, y2, indice in botones:
        if x1 <= x <= x2 and y1 <= y_barra <= y2:

            if indice == 0:
                guardar_imagen(
                    frame, CARPETA_FOTOS, "foto_original"
                )

            elif indice == 1:
                resultado = dibujar_detecciones(frame, detecciones)

                guardar_imagen(
                    resultado, CARPETA_RESULTADOS, "resultado_yolo"
                )

            elif indice == 2:
                if not grabacion_activa:
                    alto, ancho = frame.shape[:2]

                    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ruta = os.path.join(
                        CARPETA_VIDEOS,
                        f"grabacion_{fecha}.avi"
                    )

                    nuevo_grabador = cv2.VideoWriter(
                        ruta,
                        cv2.VideoWriter_fourcc(*"XVID"),
                        15.0,
                        (ancho, alto)
                    )

                    if nuevo_grabador.isOpened():
                        grabador = nuevo_grabador
                        ruta_grabacion = ruta
                        grabacion_activa = True
                        print(f"Grabacion iniciada: {ruta}")
                    else:
                        nuevo_grabador.release()
                        print("No se pudo iniciar la grabacion.")

                else:
                    grabacion_activa = False

                    if grabador is not None:
                        grabador.release()
                        grabador = None

                    print(f"Grabacion guardada: {ruta_grabacion}")

            elif indice == 3:
                with lock:
                    estado["deteccion_activa"] = (
                        not estado["deteccion_activa"]
                    )

                    if not estado["deteccion_activa"]:
                        estado["detecciones"] = []
                        estado["cantidad"] = 0

            elif indice == 4:
                estado["activo"] = False


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    global grabador, grabacion_activa
    global ultimo_original, ultimo_procesado

    if not os.path.isfile(MODELO_PATH):
        print(f"No se encuentra el modelo: {MODELO_PATH}")
        return

    print("Cargando YOLO...")
    modelo = YOLO(MODELO_PATH)

    hilo_video = threading.Thread(
        target=recibir_video,
        daemon=True
    )

    hilo_yolo = threading.Thread(
        target=ejecutar_deteccion,
        args=(modelo,),
        daemon=True
    )

    hilo_video.start()
    hilo_yolo.start()

    cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(NOMBRE_VENTANA, manejar_clic)

    # Mostrar la ventana sin forzar una resolucion mayor que el video.
    cv2.resizeWindow(NOMBRE_VENTANA, 960, 650)

    print("Video iniciado.")
    print("Q/ESC: salir | S: guardar foto | R: guardar deteccion")

    try:
        while estado["activo"]:

            with lock:  
                frame = estado["frame"]
                detecciones = list(estado["detecciones"])
                cantidad = estado["cantidad"]
                conectado = estado["conectado"]
                deteccion_activa = estado["deteccion_activa"]

                if frame is not None:
                    frame = frame.copy()

            if frame is None:
                espera = np.zeros((240, 640, 3), dtype=np.uint8)

                cv2.putText(
                    espera,
                    "Esperando ESP32-CAM...",
                    (35, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.imshow(NOMBRE_VENTANA, espera)

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    estado["activo"] = False
                    break

                continue

            # Visualizacion independiente de la inferencia.
            procesado = dibujar_detecciones(frame, detecciones)

            ultimo_original = frame
            ultimo_procesado = procesado

            # Grabacion sin detener el hilo de video.
            if grabacion_activa and grabador is not None:
                grabador.write(procesado)

            pantalla = crear_interfaz(
                procesado,
                cantidad,
                conectado,
                deteccion_activa
            )

            cv2.imshow(NOMBRE_VENTANA, pantalla)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla in (ord("q"), 27):
                estado["activo"] = False
                break

            elif tecla == ord("s"):
                guardar_imagen(
                    ultimo_original,
                    CARPETA_FOTOS,
                    "foto_original"
                )

            elif tecla == ord("r"):
                guardar_imagen(
                    ultimo_procesado,
                    CARPETA_RESULTADOS,
                    "resultado_yolo"
                )

    except KeyboardInterrupt:
        print("Programa interrumpido.")

    finally:
        estado["activo"] = False

        if grabador is not None:
            grabador.release()
            grabador = None

        cv2.destroyAllWindows()
        print("Programa finalizado.")


if __name__ == "__main__":
    main()
