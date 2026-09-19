import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

# =========================
# CONFIGURACIÓN
# =========================

ESP32_IP = "192.168.51.164"  # CAMBIAR IP DEL ESP32-CAM 

URL = f"http://{ESP32_IP}/capture"

# Cargar modelo YOLO
modelo = YOLO("yolo11l.pt")


# =========================
# CAPTURAR FOTO
# =========================

print("Solicitando fotografía al ESP32-CAM...")

respuesta = requests.get(URL)

if respuesta.status_code != 200:
    print("Error al obtener la fotografía")
    exit()

print("Fotografía recibida")


# =========================
# GUARDAR FOTO
# =========================

imagen = Image.open(BytesIO(respuesta.content))

imagen.save("foto_actual.jpg")

print("Foto guardada como foto_actual.jpg")


# =========================
# DETECTAR OBJETOS
# =========================

print("\nAnalizando imagen...\n")

resultados = modelo("foto_actual.jpg", conf =0.30)

for resultado in resultados:

    resultado.show()

    if resultado.boxes is None:
        print("No se detectaron objetos")
        continue

    for objeto in resultado.boxes:

        clase = int(objeto.cls[0])
        confianza = float(objeto.conf[0])

        nombre = modelo.names[clase]

        print(
            f"Objeto: {nombre} | "
            f"Confianza: {confianza:.2%}"
        )