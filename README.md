# ESP32-CAM + YOLO

## Descripción

Proyecto de visión por computadora utilizando una ESP32-CAM y YOLO.

La ESP32-CAM captura una fotografía y la envía mediante WiFi al computador a través de HTTP. 
Luego, Python recibe la imagen y utiliza un modelo YOLO para detectar diferentes objetos.

El proyecto está pensado como una actividad didáctica para alumnos de 4to y 5to de secundaria,
permitiendo experimentar con IoT, cámaras, Python e inteligencia artificial.

El modelo puede detectar diferentes objetos cotidianos, no solamente sillas, incluyendo frutas,
plantas en maceta, botellas, libros, celulares, teclados y otros objetos.

Los resultados se muestran en español y las imágenes procesadas se guardan para poder revisar las
detecciones realizadas por YOLO.

---

## Requisitos

- Python 3
- ESP32-CAM
- Conexión WiFi
- Arduino IDE (para cargar el programa a la ESP32-CAM)

---

## Instalación

Instalar las librerías necesarias:

```bash
pip install requests pillow ultralytics
````

El modelo YOLO utilizado es:

```text
yolo11n.pt
```

También se puede utilizar otro modelo compatible, por ejemplo:

```text
yolo11l.pt
```

---

## Configuración

Antes de ejecutar el programa, modificar en `deteccion_es.py` la IP de la ESP32-CAM:

```python
ESP32_IP = "192.168.51.164"
```

La IP puede variar según la red WiFi. Normalmente puede cambiar el último octeto.

## Ejecución

Ejecutar:

```bash
python deteccion_es.py
```
Asegurarse de estar en el directorio en donde se guardó el código .py

El programa:

1. Solicita una fotografía a la ESP32-CAM.
2. Guarda la imagen original.
3. Analiza la imagen con YOLO.
4. Muestra los objetos detectados y su nivel de confianza.
5. Guarda una imagen con las detecciones realizadas.

---

## Carpetas generadas

```text
fotos/
```

Contiene las imágenes originales capturadas por la ESP32-CAM.

```text
resultados_yolo/
```

Contiene las imágenes después del procesamiento de YOLO, con los objetos detectados y sus etiquetas.

---

## Ejemplo

```text
Objeto: silla | Confianza: 81.83%
Objeto: persona | Confianza: 79.58%
Objeto: silla | Confianza: 68.52%
Objeto: persona | Confianza: 64.28%
Objeto: persona | Confianza: 61.48%
Objeto: silla | Confianza: 57.75%
Objeto: silla | Confianza: 52.00%
Objeto: computadora portátil | Confianza: 50.20%
Objeto: persona | Confianza: 47.18%
Objeto: computadora portátil | Confianza: 46.30%
Objeto: persona | Confianza: 43.30%
Objeto: persona | Confianza: 42.13%
Objeto: mesa de comedor | Confianza: 40.40%
Objeto: computadora portátil | Confianza: 39.25%
Objeto: computadora portátil | Confianza: 38.03%
Objeto: persona | Confianza: 33.98%
Objeto: persona | Confianza: 32.40%
Objeto: computadora portátil | Confianza: 31.78%
Objeto: persona | Confianza: 31.02%
Objeto: persona | Confianza: 30.16%
```
<p align="center">
  <img width="480" height="320" alt="resultado_20260919_103744_471063" src="https://github.com/user-attachments/assets/bf45aea8-e4de-4960-b297-bafa494c4841" />
</p>
---

## Tecnologías

* ESP32-CAM
* Python
* WiFi
* HTTP
* YOLO
* Ultralytics
* Pillow
