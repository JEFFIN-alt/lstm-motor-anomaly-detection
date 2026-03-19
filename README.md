***LSTM-Based Motor Anomaly Detection System**
*** Overview

This project implements a real-time motor anomaly detection system using:

LSTM (Long Short-Term Memory)

Arduino-based sensor data collection

Python-based monitoring dashboard

The system detects abnormal behavior in motors using time-series data.

⚙️ Hardware Setup

Arduino board

Breadboard

3 Sensors (e.g., vibration, temperature, current)

Jumper wires

 Software Stack

Python

TensorFlow / Keras

NumPy, Pandas

Serial Communication (Arduino → Python)

 Workflow

Sensor data collected using Arduino

Data streamed to Python via serial port

Data preprocessed and scaled

LSTM model predicts normal behavior

Anomalies detected using thresholding

Real-time monitoring via dashboard

 Project Structure
.
├── app.py
├── motor_monitor.py
├── real_time_detector.py
├── collect_dataset.py
├── compute_threshold.py
├── ui_detector.py
├── arduino_stream_test.py
├── motor_anomaly_lstm.keras
├── scaler.save
├── requirements.txt
▶ How to Run
pip install -r requirements.txt
python motor_monitor.py
==== Features==================

Real-time anomaly detection

Hardware + AI integration

LSTM-based time series prediction

Dashboard visualization

****Future Improvements****

Deploy as web app

Improve model accuracy

Add more sensors


## 🛠 Hardware Setup
![Setup](assets/setup.jpg)

## 📊 Output
![Output](assets/output.png)