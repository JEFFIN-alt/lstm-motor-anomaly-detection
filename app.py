"""
app.py  —  Flask back-end for the Motor Anomaly Monitor
Run:  python app.py
Then open:  http://localhost:5000
"""
import json
import threading
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd
import serial
from flask import Flask, Response, render_template

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION  (keep in sync with real_time_detector.py)
# ─────────────────────────────────────────────────────────────────
COM_PORT             = "COM7"
BAUD_RATE            = 9600

SEQUENCE_LEN         = 30
FEATURE_COLS         = ["ax", "ay", "az", "gx", "gy", "gz", "temp", "flame"]
SENSOR_IDX           = 7

LSTM_THRESHOLD       = 0.0628
VIBRATION_THRESHOLD  = 60000
VIBRATION_STD_THRESH = 20000
VIBRATION_WINDOW     = 10
VIB_WEIGHT           = 0.4
LSTM_WEIGHT          = 0.6

HIST_LEN             = 200   # samples kept in history buffers

# ─────────────────────────────────────────────────────────────────
#  LOAD MODEL & SCALER
# ─────────────────────────────────────────────────────────────────
print("Loading model and scaler …")
from tensorflow.keras.models import load_model          # noqa: E402
model  = load_model("motor_anomaly_lstm.keras")
scaler = joblib.load("scaler.save")
model.predict(np.zeros((1, SEQUENCE_LEN, len(FEATURE_COLS))), verbose=0)
print("Model ready.")

# ─────────────────────────────────────────────────────────────────
#  SERIAL CONNECTION
# ─────────────────────────────────────────────────────────────────
print(f"Connecting to {COM_PORT} …")
ser = None
for attempt in range(1, 4):
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1, write_timeout=1)
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        time.sleep(0.5)
        print(f"  Connected on attempt {attempt}.")
        break
    except serial.SerialException as e:
        print(f"  Attempt {attempt}/3 failed: {e}")
        if attempt < 3:
            time.sleep(2)
        else:
            raise SystemExit("Cannot open serial port — close Arduino IDE Serial Monitor first.")

# ── Wait for Arduino to finish rebooting (DTR reset) then flush garbage ──
print("  Waiting 2 s for Arduino boot…")
time.sleep(2)
try:
    ser.reset_input_buffer()
except Exception:
    pass
print("  Buffer flushed — ready.")

ser.write(b'0')   # boot green

# ─────────────────────────────────────────────────────────────────
#  SHARED STATE  (written by detector thread, read by SSE)
# ─────────────────────────────────────────────────────────────────
_lock = threading.Lock()

_state = {
    "values"    : [0.0] * len(FEATURE_COLS),
    "mse"       : 0.0,
    "vib"       : 0.0,
    "combined"  : 0.0,
    "status"    : "STARTING",   # STARTING | BUFFERING | NORMAL | ANOMALY | CRITICAL
    "led"       : "G",
    "buf_count" : 0,
}

_mse_hist   = deque(maxlen=HIST_LEN)
_score_hist = deque(maxlen=HIST_LEN)
_vib_hist   = deque(maxlen=HIST_LEN)
_event_log  = deque(maxlen=40)   # most-recent first

# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────
def _set_led(new: str):
    if _state["led"] == new:
        return
    _state["led"] = new
    try:
        ser.write({"B": b"2", "R": b"1", "G": b"0"}[new])
    except Exception:
        pass

def _log(msg: str, kind: str):
    ts = time.strftime("%H:%M:%S")
    with _lock:
        _event_log.appendleft({"ts": ts, "msg": msg, "kind": kind})

# ─────────────────────────────────────────────────────────────────
#  DETECTOR THREAD  — mirrors real_time_detector.py logic exactly
# ─────────────────────────────────────────────────────────────────
_sequence   = deque(maxlen=SEQUENCE_LEN)
_accel_hist = deque(maxlen=VIBRATION_WINDOW)


def _detector():
    while True:
        # --- read ---
        try:
            raw = ser.readline()
        except serial.SerialException:
            time.sleep(0.5)
            continue

        if not raw:
            continue

        try:
            line = raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            continue

        parts = line.split(",")
        if len(parts) != len(FEATURE_COLS):
            continue

        try:
            values = [float(v) for v in parts]
        except ValueError:
            continue

        ax, ay, az = values[0], values[1], values[2]
        sensor_val = int(values[SENSOR_IDX])

        # --- vibration ---
        accel_mag = (ax**2 + ay**2 + az**2) ** 0.5
        _accel_hist.append(accel_mag)

        vib_spike = min(accel_mag / VIBRATION_THRESHOLD, 1.0)

        vib_std_score = 0.0
        if len(_accel_hist) == VIBRATION_WINDOW:
            vib_std       = float(np.std(_accel_hist))
            vib_std_score = min(vib_std / VIBRATION_STD_THRESH, 1.0)

        vib_score = max(vib_spike, vib_std_score)

        with _lock:
            _state["values"] = values
            _state["vib"]    = accel_mag
            _vib_hist.append(accel_mag)

        # --- sensor = 0 (instant threshold breach) ---
        if sensor_val == 0:
            raw_df = pd.DataFrame([values], columns=FEATURE_COLS)
            scaled = scaler.transform(raw_df)[0]
            _sequence.append(scaled)

            with _lock:
                if _state["status"] != "CRITICAL":
                    _log("Threshold breach detected — anomaly", "critical")
                _set_led("B")
                _state["status"]   = "CRITICAL"
                _state["combined"] = 1.0
                _score_hist.append(1.0)
            continue

        # --- scale + buffer ---
        raw_df = pd.DataFrame([values], columns=FEATURE_COLS)
        scaled = scaler.transform(raw_df)[0]
        _sequence.append(scaled)

        buf_now = len(_sequence)
        with _lock:
            _state["buf_count"] = buf_now
            if buf_now < SEQUENCE_LEN:
                _state["status"] = "BUFFERING"
        if buf_now < SEQUENCE_LEN:
            continue

        # --- LSTM inference ---
        X     = np.array(_sequence, dtype=np.float32).reshape(1, SEQUENCE_LEN, len(FEATURE_COLS))
        recon = model.predict(X, verbose=0)
        mse   = float(np.mean(np.square(X - recon)))

        lstm_score    = min(mse / LSTM_THRESHOLD, 1.0)
        combined      = LSTM_WEIGHT * lstm_score + VIB_WEIGHT * vib_score
        anomaly_cycle = combined >= 1.0 or vib_spike >= 1.0 or vib_std_score >= 1.0

        with _lock:
            _state["mse"]      = mse
            _state["combined"] = combined
            _mse_hist.append(mse)
            _score_hist.append(combined)
            prev = _state["status"]

            if anomaly_cycle:
                _set_led("R")
                _state["status"] = "ANOMALY"
                if prev != "ANOMALY":
                    _log(f"Anomaly detected  score={combined:.3f}  MSE={mse:.5f}", "anomaly")
            else:
                if _state["led"] != "B":
                    _set_led("G")
                    if prev not in ("NORMAL", "STARTING", "BUFFERING"):
                        _log(f"Returned to normal  combined={combined:.3f}", "normal")
                    _state["status"] = "NORMAL"


threading.Thread(target=_detector, daemon=True, name="detector").start()

# ─────────────────────────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        lstm_threshold=LSTM_THRESHOLD,
        vib_threshold=VIBRATION_THRESHOLD,
        sequence_len=SEQUENCE_LEN,
    )


@app.route("/stream")
def stream():
    """Server-Sent Events — pushes a JSON frame every 150 ms."""
    def _generate():
        while True:
            with _lock:
                payload = {
                    "status"     : _state["status"],
                    "mse"        : round(_state["mse"],      6),
                    "vib"        : round(_state["vib"],      1),
                    "combined"   : round(_state["combined"], 4),
                    "values"     : [round(v, 2) for v in _state["values"]],
                    "buf_count"  : _state["buf_count"],
                    # send last 120 points to keep payload small
                    "mse_hist"   : [round(v, 6) for v in list(_mse_hist)[-120:]],
                    "score_hist" : [round(v, 4) for v in list(_score_hist)[-120:]],
                    "vib_hist"   : [round(v, 0) for v in list(_vib_hist)[-120:]],
                    "log"        : list(_event_log)[:12],
                }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.15)

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control"    : "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/health")
def health():
    with _lock:
        return {"status": _state["status"], "mse": _state["mse"]}


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nDashboard -> http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
