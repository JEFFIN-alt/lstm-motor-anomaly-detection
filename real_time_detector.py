import serial
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# load trained model
model = load_model("motor_anomaly_lstm.keras")

# load scaler
scaler = joblib.load("scaler.save")

# anomaly threshold
threshold = 0.0628

# connect to Arduino
ser = serial.Serial("COM7", 9600)

sequence = []

print("Monitoring started...")

while True:

    line = ser.readline().decode().strip()

    values = line.split(",")

    if len(values) != 8:
        continue

    values = np.array(values, dtype=float)

    # scale like training
    values = scaler.transform(np.array([values]))[0]

    sequence.append(values)

    if len(sequence) > 30:
        sequence.pop(0)

    if len(sequence) == 30:

        X = np.array(sequence).reshape(1, 30, 8)

        reconstruction = model.predict(X, verbose=0)

        mse = np.mean(np.square(X - reconstruction))

        print("Error:", mse)

        if mse > threshold:
            ser.write(b'1')
            print("ANOMALY DETECTED")
        else:
            ser.write(b'0')
            print("NORMAL")