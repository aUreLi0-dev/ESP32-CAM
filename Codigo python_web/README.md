# Experimento 2: detección en tiempo real en una página web

La ESP32-CAM envía video continuo a la PC. Python analiza cada imagen con **YOLO11n**
y levanta una **página web en la misma PC** (`http://localhost:5000`), que se abre sola en el navegador.

```
ESP32-CAM  --Wi-Fi (video)-->  PC: Python + YOLO + página web (localhost:5000)
http://IP-ESP32:81/stream
```

> El código Arduino **no cambia**: el `CameraWebServer_proyecto` ya publica el video en `:81/stream`.

## Archivos

```
Código python_web/
├── servidor_web.py      # Servidor: recibe el video, ejecuta YOLO y sirve la página
├── templates/
│   └── index.html       # La página web que ven los alumnos
└── yolo11n.pt           # (copiarlo desde "Código python"; si no está, se descarga solo)
```

## Pasos

1. Instalar librerías (una sola vez):
   ```bash
   pip install ultralytics flask requests
   ```
2. En `servidor_web.py` cambiar la IP:
   ```python
   ESP32_IP = "192.168.51.22"   # IP que muestra el Monitor Serial
   ```
3. Ejecutar:
   ```bash
   python servidor_web.py
   ```
4. El navegador se abre solo en `http://localhost:5000`. Si no se abre, escribir esa dirección en Chrome o Edge.

**Opcional, compartir con celulares:** cambiar `COMPARTIR_EN_RED = True`, instalar `pip install qrcode`
y permitir Python en el Firewall de Windows. La terminal mostrará la dirección para la red (ej. `http://192.168.51.40:5000`).

## ¿Qué tiene la página?

- Video en vivo con las cajas de YOLO
- Lista de objetos detectados con su % de confianza
- Slider para cambiar la confianza mínima en vivo
- Reto: ¿cuántos de los 80 objetos de COCO pueden encontrar?
- Información del modelo: YOLO11n (Ultralytics, 2024), entrenado con COCO
- Botón para guardar la foto en la PC (carpetas `fotos/` y `resultados_yolo/`)
- (Solo con `COMPARTIR_EN_RED = True`) código QR para entrar desde el celular

## Probar sin ESP32

En `servidor_web.py` cambiar `FUENTE = "webcam"` para usar la cámara de la laptop.

## Problemas frecuentes

| Problema | Solución |
|---|---|
| La página se queda en "Esperando el video" | Recargar con Ctrl + F5; revisar que la terminal diga "ESP32-CAM conectada" |
| "Cámara desconectada" | Revisar `ESP32_IP`; presionar RST en la ESP32 |
| Video congelado | Cerrar pestañas abiertas en la IP de la ESP32 (solo admite un video a la vez) |
| Video lento | Dejar `TAMANO_IMAGEN = 320` |
| `Address already in use` | Ya hay un servidor abierto: cerrarlo con Ctrl + C o cambiar `PUERTO_WEB` |

## Sobre el modelo

- Modelo: **YOLO11n** de Ultralytics — https://docs.ultralytics.com/models/yolo11/
- Código: https://github.com/ultralytics/ultralytics
- Dataset de entrenamiento: **COCO** (80 clases) — https://cocodataset.org
- Paper original de YOLO (Redmon et al., 2016): https://arxiv.org/abs/1506.02640
