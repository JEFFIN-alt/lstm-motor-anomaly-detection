````markdown
# ⚙️ LSTM-Based Encoder–Decoder for Multi-Sensor Anomaly Detection

> Real-time AI-powered motor health monitoring using **LSTM Neural Networks**, **Arduino**, **MPU6050**, and **Python Dashboard**.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow)
![Arduino](https://img.shields.io/badge/Arduino-Hardware-00979D?logo=arduino)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project is an **Intelligent Motor Anomaly Detection System** that continuously monitors motor health using sensor data collected from an Arduino.

Sensor readings are analyzed using a trained **LSTM Autoencoder**, which learns normal operating patterns and detects abnormal behavior in real time.

When an anomaly is detected:

- 🚨 Dashboard updates instantly
- 🔴 Arduino LED turns RED
- ⚠ Critical conditions trigger blinking alerts
- 📈 Live graphs visualize sensor values and anomaly scores

---

## ✨ Features

- 🤖 LSTM Autoencoder based anomaly detection
- ⚡ Real-time sensor streaming via Serial Communication
- 📊 Live monitoring dashboard
- 📈 Reconstruction error visualization
- 🔴 Automatic LED alerts using Arduino
- 🌡 Temperature monitoring
- 📡 MPU6050 Accelerometer & Gyroscope integration
- 🎯 Threshold-based anomaly detection
- 💾 Dataset collection utility
- 🌐 Flask Web Dashboard

---

## 🛠 Hardware Used

- Arduino Uno/Nano
- MPU6050 Accelerometer & Gyroscope
- Flame Sensor (Digital)
- LEDs (Red & Green)
- Breadboard
- Jumper Wires
- USB Cable

---

## 💻 Software Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Flask
- Streamlit
- PySerial
- Plotly

---

## ⚙️ System Architecture

```text
               +--------------------+
               |     Arduino        |
               | MPU6050 + Sensors  |
               +---------+----------+
                         |
                  Serial Communication
                         |
                         ▼
               +--------------------+
               |   Python Backend   |
               | Data Preprocessing |
               +---------+----------+
                         |
                  StandardScaler
                         |
                         ▼
             +----------------------+
             |  LSTM Autoencoder    |
             +----------+-----------+
                        |
        Reconstruction Error Calculation
                        |
               Threshold Comparison
                        |
        +---------------+----------------+
        |                                |
     NORMAL                        ANOMALY
        |                                |
 Green LED ON                  Red LED ON
        |                                |
        +----------Dashboard-------------+
```

---

# 📂 Project Structure

```text
jeffin-alt-lstm-motor-anomaly-detection/
│
├── app.py                      # Flask web application
├── dashboard.py                # Streamlit monitoring dashboard
├── motor_monitor.py            # Main monitoring script
├── real_time_detector.py       # Real-time anomaly detection
├── ui_detector.py              # UI detector
├── collect_dataset.py          # Dataset collection
├── compute_threshold.py        # Threshold calculation
├── arduino_stream_test.py      # Serial communication testing
├── arduino_led_control.ino     # Arduino firmware
├── motor_normal_data.csv       # Sample dataset
├── scaler.save                 # Saved scaler
├── templates/
│   └── index.html              # Flask frontend
│
└── assets/
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/jeffin-alt-lstm-motor-anomaly-detection.git

cd jeffin-alt-lstm-motor-anomaly-detection
```

Install dependencies

```bash
pip install -r requirement.txt
```

---

# ▶ Running the Project

### 1. Upload Arduino Code

Upload

```text
arduino_led_control.ino
```

to your Arduino board.

---

### 2. Collect Dataset (Optional)

```bash
python collect_dataset.py
```

---

### 3. Compute Threshold

```bash
python compute_threshold.py
```

---

### 4. Start Real-Time Monitoring

```bash
python motor_monitor.py
```

or

```bash
python real_time_detector.py
```

---

### 5. Launch Dashboard

Flask Dashboard

```bash
python app.py
```

Open

```
http://localhost:5000
```

or

Streamlit Dashboard

```bash
streamlit run dashboard.py
```

---

# 📊 Detection Workflow

1. Arduino collects sensor values.
2. Sensor data is streamed to Python.
3. Data is normalized using the saved scaler.
4. LSTM predicts reconstructed sensor sequence.
5. Reconstruction error is calculated.
6. Error is compared with threshold.
7. Dashboard updates in real time.
8. Arduino LEDs indicate motor health.

---

# 📈 Model

Model Type:

- LSTM Autoencoder

Input Features:

- Accelerometer X
- Accelerometer Y
- Accelerometer Z
- Gyroscope X
- Gyroscope Y
- Gyroscope Z
- Temperature
- Flame Sensor

Sequence Length

```
30 Timesteps
```

Detection Metric

```
Mean Squared Reconstruction Error
```

---

# 📸 Screenshots

## Hardware Setup

```
assets/setup.jpg
```

## Dashboard

```
assets/output.png
```

---

# 🔮 Future Improvements

- IoT Cloud Integration
- Email & SMS Alerts
- Mobile Application
- MQTT Support
- Edge AI Deployment
- Predictive Maintenance Analytics
- Multiple Motor Support
- Model Optimization

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Thejus P K**

If you found this project helpful, consider giving it a ⭐ on GitHub!
````
