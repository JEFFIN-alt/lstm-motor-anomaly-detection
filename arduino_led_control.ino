/*
  ============================================================
  Motor Anomaly Detector — Arduino Sketch
  ============================================================
  WIRING:
    MPU6050:
      VCC  → 3.3V  |  GND → GND  |  SDA → A4  |  SCL → A5

    Sensor (Analog/Digital):
      VCC → 5V  |  GND → GND  |  DO → Pin 7
      (DO = LOW/0 = threshold triggered, HIGH/1 = normal)

    RED   LED → Pin 9  → 220Ω resistor → GND
    GREEN LED → Pin 8  → 220Ω resistor → GND

  Baud rate: 9600   (must match BAUD_RATE in real_time_detector.py)
  Sends every 50ms:  ax,ay,az,gx,gy,gz,temp,sensor
  Receives:
    '0' = NORMAL       → GREEN on,  RED off
    '1' = ANOMALY      → RED steady, GREEN off
    '2' = CRITICAL     → RED blink,  GREEN off
  ============================================================
*/

#include <Wire.h>

#define SENSOR_PIN 7
#define RED_LED 9   // ← RED  is on Pin 9
#define GREEN_LED 8 // ← GREEN is on Pin 8

#define MPU6050_ADDR 0x68

int16_t ax, ay, az, gx, gy, gz, rawTemp;

// Blink state (non-blocking)
bool blinkMode = false;
bool redSteady = false;
bool blinkLedState = false;
unsigned long lastBlink = 0;
#define BLINK_INTERVAL 200 // ms between toggles

void initMPU6050() {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1
  Wire.write(0x00); // Wake up
  Wire.endTransmission(true);
  delay(100);
}

void readMPU6050() {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 14, true);
  ax = (Wire.read() << 8) | Wire.read();
  ay = (Wire.read() << 8) | Wire.read();
  az = (Wire.read() << 8) | Wire.read();
  rawTemp = (Wire.read() << 8) | Wire.read();
  gx = (Wire.read() << 8) | Wire.read();
  gy = (Wire.read() << 8) | Wire.read();
  gz = (Wire.read() << 8) | Wire.read();
}

void applyLedState() {
  if (blinkMode) {
    // Non-blocking blink
    unsigned long now = millis();
    if (now - lastBlink >= BLINK_INTERVAL) {
      lastBlink = now;
      blinkLedState = !blinkLedState;
      digitalWrite(RED_LED, blinkLedState ? HIGH : LOW);
      digitalWrite(GREEN_LED, LOW);
    }
  } else if (redSteady) {
    digitalWrite(RED_LED, HIGH);
    digitalWrite(GREEN_LED, LOW);
  } else {
    digitalWrite(RED_LED, LOW);
    digitalWrite(GREEN_LED, HIGH);
  }
}

void setup() {
  Serial.begin(9600);
  Wire.begin();
  initMPU6050();

  pinMode(SENSOR_PIN, INPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);

  // Boot state: GREEN on (system normal)
  digitalWrite(RED_LED, LOW);
  digitalWrite(GREEN_LED, HIGH);

  delay(500);
}

void loop() {
  // ── Receive LED command from Python (check FIRST for responsiveness) ──
  while (Serial.available() > 0) {
    char cmd = (char)Serial.read();
    if (cmd == '2') {
      // Critical threshold crossed → RED blink
      blinkMode = true;
      redSteady = false;
    } else if (cmd == '1') {
      // Anomaly (LSTM/vibration) → RED steady
      blinkMode = false;
      redSteady = true;
    } else if (cmd == '0') {
      // Normal → GREEN on
      blinkMode = false;
      redSteady = false;
    }
  }

  // ── Apply LED state (runs every loop — handles blink timing) ──
  applyLedState();

  // ── Read sensors ──────────────────────────────────────────
  readMPU6050();
  int sensorVal = digitalRead(SENSOR_PIN); // 0 = triggered, 1 = normal
  int temp = (int)((rawTemp / 340.0) + 36.53);

  // ── Send CSV to Python ────────────────────────────────────
  // Format: ax,ay,az,gx,gy,gz,temp,sensor
  Serial.print(ax);
  Serial.print(",");
  Serial.print(ay);
  Serial.print(",");
  Serial.print(az);
  Serial.print(",");
  Serial.print(gx);
  Serial.print(",");
  Serial.print(gy);
  Serial.print(",");
  Serial.print(gz);
  Serial.print(",");
  Serial.print(temp);
  Serial.print(",");
  Serial.println(sensorVal); // println adds \n — required for Python readline()

  delay(50); // 50ms = ~20 readings/second
}
