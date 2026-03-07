import serial
import numpy as np
import pandas as pd
import joblib
import time
from collections import deque
from tensorflow.keras.models import load_model

# ================================================================
#  CONFIGURATION  — adjust these if needed
# ================================================================
COM_PORT       = "COM7"
BAUD_RATE      = 9600       # Must match Serial.begin() in Arduino sketch

SEQUENCE_LEN   = 30         # Fixed by trained model — do not change
FEATURE_COLS   = ["ax","ay","az","gx","gy","gz","temp","flame"]  # "flame" kept to match scaler fit
SENSOR_IDX     = 7          # digital sensor is the last (8th) column
ACCEL_IDX      = [0, 1, 2]  # ax, ay, az indices

# ── Unified anomaly threshold ───────────────────────────────────
# A combined score is computed from:
#   1. LSTM reconstruction error (MSE)       — catches subtle patterns
#   2. Vibration magnitude (raw accel units) — catches sudden spikes
#   3. Vibration std-dev (rolling window)    — catches sustained abnormal vibration
#   4. Digital sensor reading (0 = triggered) — instant threshold breach
#
# When the combined score exceeds ANOMALY_THRESHOLD → Anomaly Detected
# When the digital sensor reads 0 (instant breach) → RED LED blinks

LSTM_THRESHOLD      = 0.0628   # Tune via compute_threshold.py

VIBRATION_THRESHOLD = 60000    # raw MPU6050 units — raised to ignore normal motor vibration
VIBRATION_WINDOW    = 10       # rolling samples for std-dev check
VIBRATION_STD_THRESH= 20000    # raised to ignore normal motor vibration

# Weights for the unified score (0–1 per component, normalized)
# Vibration score weight (relative to LSTM)
VIB_WEIGHT          = 0.4      # how much vibration contributes to combined score
LSTM_WEIGHT         = 0.6      # how much LSTM MSE contributes to combined score

# ================================================================
#  Load model & scaler
# ================================================================
print("Loading model and scaler...")
model  = load_model("motor_anomaly_lstm.keras")
scaler = joblib.load("scaler.save")
model.predict(np.zeros((1, SEQUENCE_LEN, len(FEATURE_COLS))), verbose=0)
print("Model ready.\n")

# ================================================================
#  Serial connection
# ================================================================
print(f"Opening {COM_PORT} at {BAUD_RATE} baud...")
ser = None
for attempt in range(1, 4):   # try up to 3 times
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1, write_timeout=1)
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        time.sleep(0.5)
        print(f"Connected on attempt {attempt}.")
        break
    except serial.SerialException as e:
        print(f"Attempt {attempt}/3 failed: {e}")
        if attempt < 3:
            print("  Make sure Arduino IDE Serial Monitor is CLOSED. Retrying in 2s...")
            time.sleep(2)
        else:
            print("\nCould not open COM port after 3 attempts.")
            print("Steps to fix:")
            print("  1. Close Arduino IDE (especially Serial Monitor)")
            print("  2. Unplug and replug the Arduino USB cable")
            print("  3. Run this script again")
            raise SystemExit(1)

# ── Startup diagnostic ─────────────────────────────────────────
print(f"Connected: {COM_PORT} @ {BAUD_RATE} baud")
print("Checking first 3 data lines from Arduino:")
good = 0
for _ in range(15):
    raw = ser.readline()
    if not raw:
        continue
    try:
        txt = raw.decode("utf-8", errors="replace").strip()
        parts = txt.split(",")
        ok = "OK" if len(parts) == len(FEATURE_COLS) else "BAD"
        print(f"  [{ok}] {txt}")
        if len(parts) == len(FEATURE_COLS):
            good += 1
        if good >= 3:
            break
    except Exception:
        continue

if good == 0:
    print("  WARNING: No valid 8-column data received. Check wiring/baud rate.")
else:
    print(f"  Data looks good ({good}/3 valid)\n")

ser.reset_input_buffer()

# ── Send initial state to Arduino right away ──────────────────
ser.write(b'0')      # Start GREEN (normal)
print("-" * 60)
print(f"LSTM threshold      : {LSTM_THRESHOLD}")
print(f"Vibration spike     : {VIBRATION_THRESHOLD}  (raw units — tune if needed)")
print(f"Vibration std       : {VIBRATION_STD_THRESH} (tune if needed)")
print(f"Weights             : LSTM={LSTM_WEIGHT}, Vibration={VIB_WEIGHT}")
print("-" * 60)
print("Monitoring started...\n")

# ================================================================
#  State
# ================================================================
sequence       = deque(maxlen=SEQUENCE_LEN)
accel_mag_hist = deque(maxlen=VIBRATION_WINDOW)
led_state      = 'G'   # 'G' = green (normal), 'R' = red steady, 'B' = red blink

def set_led(state: str, reason: str):
    """
    state: 'G' = green (normal)
           'R' = red steady (anomaly)
           'B' = red blink (critical threshold breach)
    Change LED only when state actually changes.
    """
    global led_state
    if state == led_state:
        return
    led_state = state
    try:
        if state == 'B':
            ser.write(b'2')     # Arduino: RED blink
            print(f"\n>>> [RED  BLINK   ] *** ANOMALY DETECTED *** — {reason}")
        elif state == 'R':
            ser.write(b'1')     # Arduino: RED steady
            print(f"\n>>> [RED  LED  ON ] *** ANOMALY DETECTED *** — {reason}")
        else:
            ser.write(b'0')     # Arduino: GREEN
            print(f"\n>>> [GREEN LED  ON] System normal — {reason}")
        print("-" * 60)
    except serial.SerialException as e:
        print(f"Serial write error: {e}")

# ================================================================
#  Main loop
# ================================================================
while True:
    # ─ 1. Read line from Arduino ─────────────────────────────
    try:
        raw = ser.readline()
    except serial.SerialException as e:
        print(f"Serial read error: {e}")
        time.sleep(0.5)
        continue

    if not raw:
        continue

    try:
        line = raw.decode("utf-8", errors="ignore").strip()
    except Exception:
        continue

    if not line:
        continue

    parts = line.split(",")
    if len(parts) != len(FEATURE_COLS):
        continue

    try:
        values = [float(v) for v in parts]
    except ValueError:
        continue

    ax, ay, az   = values[0], values[1], values[2]
    sensor_val   = int(values[SENSOR_IDX])   # 0 = threshold breached, 1 = normal

    # ─ 2. Vibration magnitude (instant, raw) ─────────────────
    accel_mag = (ax**2 + ay**2 + az**2) ** 0.5
    accel_mag_hist.append(accel_mag)

    # Normalised vibration score [0..1]
    vib_spike_score = min(accel_mag / VIBRATION_THRESHOLD, 1.0)

    vib_std_score = 0.0
    if len(accel_mag_hist) == VIBRATION_WINDOW:
        vib_std = float(np.std(accel_mag_hist))
        vib_std_score = min(vib_std / VIBRATION_STD_THRESH, 1.0)

    # Combined vibration sub-score (take the worse of the two)
    vib_score = max(vib_spike_score, vib_std_score)

    # ─ 3. Digital sensor instant check ───────────────────────
    # sensor_val == 0 means threshold breached on hardware level
    if sensor_val == 0:
        print(f"[SENSOR=0] Threshold breached on sensor input → ANOMALY DETECTED (blink)")
        set_led('B', "Sensor threshold breach")
        # Keep buffer in sync but skip LSTM this cycle
        raw_df = pd.DataFrame([values], columns=FEATURE_COLS)
        scaled = scaler.transform(raw_df)[0]
        sequence.append(scaled)
        continue

    # ─ 4. Vibration-only instant check ───────────────────────
    anomaly_this_cycle = False   # track if ANY check fires this iteration

    if vib_spike_score >= 1.0:
        print(f"[VIBRATION] Spike mag={accel_mag:.0f} > {VIBRATION_THRESHOLD} → ANOMALY DETECTED")
        set_led('R', f"Vibration spike mag={accel_mag:.0f}")
        anomaly_this_cycle = True

    elif vib_std_score >= 1.0:
        print(f"[VIBRATION] Std-dev exceeded → ANOMALY DETECTED")
        set_led('R', "Vibration std-dev exceeded")
        anomaly_this_cycle = True

    # ─ 5. LSTM — model-based detection ────────────────────────
    raw_df = pd.DataFrame([values], columns=FEATURE_COLS)
    scaled = scaler.transform(raw_df)[0]
    sequence.append(scaled)

    if len(sequence) < SEQUENCE_LEN:
        print(f"[BUFFERING] {len(sequence)}/{SEQUENCE_LEN}  |  "
              f"sensor={sensor_val}  accel_mag={accel_mag:.0f}  vib={vib_score:.2f}")
        continue

    X     = np.array(sequence, dtype=np.float32).reshape(1, SEQUENCE_LEN, len(FEATURE_COLS))
    recon = model.predict(X, verbose=0)
    mse   = float(np.mean(np.square(X - recon)))

    # Normalised LSTM score [0..1]
    lstm_score = min(mse / LSTM_THRESHOLD, 1.0)

    # ─ 6. Unified combined anomaly score ─────────────────────
    combined_score = (LSTM_WEIGHT * lstm_score) + (VIB_WEIGHT * vib_score)
    # combined_score >= 1.0 means at least one component fully exceeded its threshold
    if combined_score >= 1.0:
        anomaly_this_cycle = True

    # ─ 7. Print every reading ─────────────────────────────────
    status = "ANOMALY" if anomaly_this_cycle else "NORMAL"
    print(f"sensor={sensor_val}  accel_mag={accel_mag:7.0f}  "
          f"MSE={mse:.5f}  vib={vib_score:.2f}  combined={combined_score:.3f}  | {status}")

    # ─ 8. Update LED ──────────────────────────────────────────
    if anomaly_this_cycle:
        # Vibration check may have already set RED; LSTM confirms or reinforces it
        set_led('R', f"Combined score={combined_score:.3f} (MSE={mse:.5f}, vib={vib_score:.2f})")
    else:
        # Only recover to green when blink-mode is NOT active
        # (blink persists until the digital sensor clears on its own)
        if led_state != 'B':
            set_led('G', f"All clear — combined={combined_score:.3f}")