import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from datetime import datetime
import os

# ============================================================
# INVENTARIO CON ESP32-CAM + YOLO
# Pensado como ejercicio didáctico para alumnos de
# 4to y 5to de secundaria.
#
# La idea es que el alumno pueda ver el flujo completo:
# cámara -> WiFi/HTTP -> Python -> YOLO -> resultado.
# ============================================================

# =========================
# CONFIGURACIÓN
# =========================

ESP32_IP = "192.168.51.164"  # CAMBIAR por la IP actual del ESP32-CAM
URL = f"http://{ESP32_IP}/capture"

# Modelo YOLO preentrenado.
# Este modelo no está limitado a sillas: trabaja con múltiples
# objetos cotidianos del conjunto COCO, incluyendo frutas,
# plantas en maceta y objetos de uso diario.
# También se puede sustituir por un modelo personalizado (.pt).
MODELO_PATH = "yolo11l.pt"
modelo = YOLO(MODELO_PATH)


# =========================
# NOMBRES EN ESPAÑOL
# =========================

# YOLO/COCO maneja internamente los nombres de las clases en inglés.
# Este diccionario permite mostrar los resultados en español.
nombres_es = {
    0: "persona",
    1: "bicicleta",
    2: "auto",
    3: "motocicleta",
    4: "avión",
    5: "autobús",
    6: "tren",
    7: "camión",
    8: "barco",
    9: "semáforo",
    10: "hidrante",
    11: "señal de pare",
    12: "parquímetro",
    13: "banco",
    14: "pájaro",
    15: "gato",
    16: "perro",
    17: "caballo",
    18: "oveja",
    19: "vaca",
    20: "elefante",
    21: "oso",
    22: "cebra",
    23: "jirafa",
    24: "mochila",
    25: "paraguas",
    26: "bolso",
    27: "corbata",
    28: "maleta",
    29: "frisbee",
    30: "esquís",
    31: "tabla de snowboard",
    32: "pelota deportiva",
    33: "cometa",
    34: "bate de béisbol",
    35: "guante de béisbol",
    36: "patineta",
    37: "tabla de surf",
    38: "raqueta de tenis",
    39: "botella",
    40: "copa de vino",
    41: "taza",
    42: "tenedor",
    43: "cuchillo",
    44: "cuchara",
    45: "tazón",
    46: "plátano",
    47: "manzana",
    48: "sándwich",
    49: "naranja",
    50: "brócoli",
    51: "zanahoria",
    52: "hot dog",
    53: "pizza",
    54: "dona",
    55: "pastel",
    56: "silla",
    57: "sofá",
    58: "planta en maceta",
    59: "cama",
    60: "mesa de comedor",
    61: "inodoro",
    62: "televisor",
    63: "computadora portátil",
    64: "mouse",
    65: "control remoto",
    66: "teclado",
    67: "celular",
    68: "microondas",
    69: "horno",
    70: "tostadora",
    71: "lavadero",
    72: "refrigeradora",
    73: "libro",
    74: "reloj",
    75: "florero",
    76: "tijeras",
    77: "oso de peluche",
    78: "secadora de cabello",
    79: "cepillo de dientes"
}


# =========================
# CREAR CARPETA PARA FOTOS
# =========================

# Todas las fotografías se guardarán dentro de esta carpeta.
CARPETA_FOTOS = "fotos"

# Si la carpeta no existe, se crea automáticamente.
os.makedirs(CARPETA_FOTOS, exist_ok=True)


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
# GUARDAR FOTO ORIGINAL
# =========================

imagen = Image.open(BytesIO(respuesta.content))

fecha_hora = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

# Carpeta para las fotos originales
CARPETA_FOTOS = "fotos"
os.makedirs(CARPETA_FOTOS, exist_ok=True)

ruta_foto = os.path.join(
    CARPETA_FOTOS,
    f"foto_{fecha_hora}.jpg"
)

imagen.save(ruta_foto)

print(f"Foto original guardada como: {ruta_foto}")


# =========================
# DETECTAR OBJETOS
# =========================

print("\nAnalizando imagen...\n")
print("El modelo puede detectar diferentes objetos cotidianos, no solo sillas.")
print("Ejemplos: frutas, plantas en maceta, botellas, libros, celulares, teclados y más.\n")

resultados = modelo(ruta_foto, conf=0.30)


# =========================
# GUARDAR RESULTADO DE YOLO
# =========================

CARPETA_RESULTADOS = "resultados_yolo"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

for resultado in resultados:

    # Mostrar nombres en español
    resultado.names = nombres_es

    if resultado.boxes is None or len(resultado.boxes) == 0:
        print("No se detectaron objetos")
        
        # Guardar igualmente la foto aunque no haya detecciones
        ruta_resultado = os.path.join(
            CARPETA_RESULTADOS,
            f"resultado_{fecha_hora}.jpg"
        )
        resultado.save(filename=ruta_resultado)

        continue

    # Guardar imagen con los resultados de YOLO
    ruta_resultado = os.path.join(
        CARPETA_RESULTADOS,
        f"resultado_{fecha_hora}.jpg"
    )

    resultado.save(filename=ruta_resultado)

    print(f"Resultado YOLO guardado como: {ruta_resultado}")

    # Mostrar resultado
    resultado.show()

    # Mostrar información en consola
    for objeto in resultado.boxes:

        clase = int(objeto.cls[0])
        confianza = float(objeto.conf[0])

        nombre = nombres_es.get(clase, modelo.names[clase])

        print(
            f"Objeto: {nombre} | "
            f"Confianza: {confianza:.2%}"
        )