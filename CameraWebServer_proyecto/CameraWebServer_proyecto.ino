#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>

// ===========================
// Select camera model
// ===========================
#include "board_config.h"

// ===========================
// WiFi
// ===========================
const char *ssid = "IoT2";
const char *password = "laboratorioIOT";

void startCameraServer();
void setupLedFlash();

void setup() {

  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // ===========================
  // Camera configuration
  // ===========================

  camera_config_t config;

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  // ===========================
  // Image settings
  // ===========================

  config.xclk_freq_hz = 20000000;

  // HVGA = 480 x 320
  config.frame_size = FRAMESIZE_HVGA;


  // JPEG is best for streaming
  config.pixel_format = PIXFORMAT_JPEG;

  // Get the newest frame instead of waiting for old frames
  config.grab_mode = CAMERA_GRAB_LATEST;

  // Use PSRAM
  config.fb_location = CAMERA_FB_IN_PSRAM;

  // Higher value = more compression = smaller image
  // 15 is a good compromise between quality and speed
  //config.jpeg_quality = 15;
  config.jpeg_quality = 12;

  // Two frame buffers improve streaming
  config.fb_count = 2;

  // ===========================
  // Camera initialization
  // ===========================

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {

    Serial.printf(
      "Camera init failed with error 0x%x\n",
      err
    );

    Serial.println(
      "La camara no pudo inicializarse."
    );

    return;
  }

  Serial.println(
    "CAMARA INICIALIZADA CORRECTAMENTE"
  );

  // ===========================
  // Sensor configuration
  // ===========================

  sensor_t *s = esp_camera_sensor_get();

  // Some sensors need vertical correction
  if (s->id.PID == OV3660_PID) {

    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);

  }

  // Force HVGA
  if (config.pixel_format == PIXFORMAT_JPEG) {

    s->set_framesize(
      s,
      FRAMESIZE_HVGA
    );

  }

  // ===========================
  // Other camera models
  // ===========================

#if defined(CAMERA_MODEL_M5STACK_WIDE) || \
    defined(CAMERA_MODEL_M5STACK_ESP32CAM)

  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);

#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)

  s->set_vflip(s, 1);

#endif

  // ===========================
  // Flash LED
  // ===========================

#if defined(LED_GPIO_NUM)

  setupLedFlash();

#endif

  // ===========================
  // WiFi
  // ===========================

  WiFi.begin(ssid, password);

  // Disable WiFi sleep to improve streaming
  WiFi.setSleep(false);

  Serial.println();
  Serial.print("WiFi connecting");

  while (WiFi.status() != WL_CONNECTED) {

    delay(500);
    Serial.print(".");

  }

  Serial.println();
  Serial.println("WiFi connected");

  // ===========================
  // Start web server
  // ===========================

  startCameraServer();

  Serial.print(
    "Camera Ready! Use 'http://"
  );

  Serial.print(
    WiFi.localIP()
  );

  Serial.println(
    "' to connect"
  );
}

void loop() {

  // The web server handles the camera stream
  delay(10000);

}