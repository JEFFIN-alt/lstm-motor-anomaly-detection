"""
app.py – Flask backend for Motor Anomaly Detection
Run: python app.py
Open: http://localhost:5000
"""

import json
import time
import threading
from collections import deque

import numpy as np
import pandas as pd
import serial
import joblib

from flask import Flask, Response, render_template
from tensorflow.keras.models import load_model

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------

COM_PORT = "COM7"
BAUD_RATE = 9600

SEQUENCE_LEN = 30

FEATURE_COLS = [
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "temp",
    "flame"
]

LSTM_THRESHOLD = 0.0628

HISTORY_LEN = 200

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

print("Loading model...")

model = load_model("motor_anomaly_lstm.keras")
scaler = joblib.load("scaler.save")

print("Model loaded successfully")

# ----------------------------------------------------
# SERIAL CONNECTION
# ----------------------------------------------------

print("Connecting to Arduino...")

ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)

time.sleep(2)

print("Connected to Arduino")

# ----------------------------------------------------
# BUFFERS
# ----------------------------------------------------

sequence_buffer = deque(maxlen=SEQUENCE_LEN)
history = deque(maxlen=HISTORY_LEN)

current_status = "NORMAL"
current_score = 0

# ----------------------------------------------------
# FLASK APP
# ----------------------------------------------------

app = Flask(__name__)


# ----------------------------------------------------
# SENSOR READING THREAD
# ----------------------------------------------------

def sensor_loop():

    global current_status
    global current_score

    while True:

        try:

            line = ser.readline().decode().strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) != len(FEATURE_COLS):
                continue

            values = [float(x) for x in parts]

            sequence_buffer.append(values)

            history.append(values)

            if len(sequence_buffer) == SEQUENCE_LEN:

                df = pd.DataFrame(sequence_buffer, columns=FEATURE_COLS)

                scaled = scaler.transform(df)

                X = np.array([scaled])

                pred = model.predict(X, verbose=0)

                error = np.mean((scaled - pred[0]) ** 2)

                current_score = float(error)

                if error > LSTM_THRESHOLD:
                    current_status = "ANOMALY"
                else:
                    current_status = "NORMAL"

        except Exception as e:

            print("Sensor error:", e)


# ----------------------------------------------------
# API STREAM
# ----------------------------------------------------

@app.route("/data")
def data():

    def generate():

        while True:

            payload = {
                "status": current_status,
                "score": current_score,
                "history": list(history)
            }

            yield f"data:{json.dumps(payload)}\n\n"

            time.sleep(0.5)

    return Response(generate(), mimetype="text/event-stream")


# ----------------------------------------------------
# DASHBOARD PAGE
# ----------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------
# START THREAD
# ----------------------------------------------------

thread = threading.Thread(target=sensor_loop)
thread.daemon = True
thread.start()

# ----------------------------------------------------
# RUN SERVER
# ----------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)