import streamlit as st
import serial
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
import joblib
import time

# -----------------------------
# Load model and scaler
# -----------------------------
model = load_model("motor_anomaly_lstm.keras")
scaler = joblib.load("scaler.save")

threshold = 0.0628
sequence_length = 30   # faster detection

# -----------------------------
# Serial connection
# -----------------------------
ser = serial.Serial("COM7", 9600)

# -----------------------------
# Streamlit layout
# -----------------------------
st.title("⚙️ Motor Health Monitoring Dashboard")

status_placeholder = st.empty()
error_placeholder = st.empty()
sensor_placeholder = st.empty()
graph_placeholder = st.empty()

data_buffer = []
errors = []

while True:

    line = ser.readline().decode().strip()
    values = line.split(",")

    if len(values) != 8:
        continue

    values = list(map(float, values))

    data_buffer.append(values)

    if len(data_buffer) > sequence_length:
        data_buffer.pop(0)

    if len(data_buffer) == sequence_length:

        input_data = scaler.transform(data_buffer)
        input_data = np.array([input_data])

        reconstruction = model.predict(input_data)

        mse = np.mean(np.square(input_data - reconstruction))

        errors.append(mse)

        if len(errors) > 100:
            errors.pop(0)

        # ------------------
        # Status
        # ------------------

        if mse > threshold:
            status_placeholder.error("🔴 ANOMALY DETECTED")
            ser.write(b'A')
        else:
            status_placeholder.success("🟢 SYSTEM NORMAL")
            ser.write(b'N')

        error_placeholder.write(f"Reconstruction Error: {mse:.4f}")

        # ------------------
        # Sensor values
        # ------------------

        ax, ay, az, gx, gy, gz, temp, flame = values

        sensor_df = pd.DataFrame({
            "Sensor":[
                "AX","AY","AZ",
                "GX","GY","GZ",
                "Temperature","Flame"
            ],
            "Value":[
                ax, ay, az,
                gx, gy, gz,
                temp, flame
            ]
        })

        sensor_placeholder.table(sensor_df)

        # ------------------
        # Error graph
        # ------------------

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=errors,
            mode="lines",
            name="Reconstruction Error"
        ))

        fig.add_hline(
            y=threshold,
            line_dash="dash",
            annotation_text="Threshold"
        )

        graph_placeholder.plotly_chart(fig, use_container_width=True)

    time.sleep(0.05)