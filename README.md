# ⚙️ LSTM-Based Multi-Sensor Anomaly Detection

> Real-time AI-powered motor health monitoring using an **LSTM Autoencoder**, **Arduino**, **MPU6050**, and an interactive **Python Dashboard** — plus Remaining Useful Life (RUL) prediction on the NASA CMAPSS benchmark dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![Arduino](https://img.shields.io/badge/Arduino-Uno%2FNano-00979D?logo=arduino&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Dashboard-black?logo=flask)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Monitor-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project implements an **Intelligent Motor Anomaly Detection System** with two complementary modules:

| Module | Description |
|---|---|
| **Real-Time Motor Monitor** | Arduino streams live multi-sensor data → LSTM Autoencoder detects anomalies → Dashboard + LED alerts |
| **NASA CMAPSS RUL Predictor** | LSTM trained on the NASA turbofan engine dataset to predict Remaining Useful Life (RUL) of industrial equipment |

The system learns what "normal" motor behaviour looks like and flags deviations in real time — enabling predictive maintenance before failures occur.

---

## ✨ Key Features

- 🤖 **LSTM Autoencoder** trained on normal sensor sequences; anomaly score = mean reconstruction error
- ⚡ **Real-time serial streaming** from Arduino via PySerial
- 📊 **Live dashboard** (Flask + Streamlit) with rolling graphs and anomaly score timeline
- 🔴 **Hardware feedback** — Arduino LEDs switch green ↔ red based on motor health
- 🌡 **Multi-sensor fusion** — accelerometer, gyroscope, temperature, and flame sensor inputs
- 🔧 **Threshold calculator** (`compute_threshold.py`) derived from validation reconstruction errors
- 🛩 **NASA CMAPSS benchmark** — LSTM-based RUL regression on public turbofan engine dataset
- 💾 Pre-trained models (`motor_anomaly_lstm.keras`, `nasa_lstm_rul_model.keras`) included

---

## 🛠 Hardware Components

| Component | Role |
|---|---|
| Arduino Uno / Nano | Microcontroller — reads sensors, controls LEDs |
| MPU6050 | 6-axis accelerometer + gyroscope (I²C) |
| Flame Sensor (digital) | Detects abnormal heat signatures |
| Red & Green LEDs | Visual anomaly indicator |
| Breadboard + Jumper Wires | Prototyping connections |
| USB Cable | Serial communication with host PC |

---

## 💻 Software Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | LSTM Autoencoder & RUL model |
| NumPy / Pandas | Data processing |
| Scikit-learn | StandardScaler, metrics |
| Joblib | Scaler serialisation |
| PySerial | Arduino serial communication |
| Flask | Web dashboard backend |
| Streamlit | Live monitoring UI |
| Plotly | Interactive charts |

---

## 🗂 Project Structure

```
LSTM-Based-Multi-Sensor-Anomaly-Detection/
│
├── arduino_led_control.ino      # Arduino firmware — reads sensors, drives LEDs
├── arduino_stream_test.py       # Verify serial connection before running monitor
│
├── collect_dataset.py           # Stream & save labelled sensor data to CSV
├── motor_normal_data.csv        # Sample dataset (normal motor operation)
├── motor_normal_labeled.csv     # Labelled version of the above
├── sensor_dataset.csv           # Extended sensor dataset
│
├── g.ipynb                      # Model training notebook (motor anomaly)
├── motor_anomaly_lstm.keras     # Trained LSTM Autoencoder (motor)
├── scaler.save                  # Fitted StandardScaler for inference
│
├── compute_threshold.py         # Computes MSE threshold from validation data
├── motor_monitor.py             # Main monitoring loop (serial → model → alert)
├── real_time_detector.py        # Lightweight real-time detector variant
├── ui_detector.py               # UI-integrated detector
│
├── dashboard.py                 # Streamlit live monitoring dashboard
├── app.py                       # Flask web app
├── templates/
│   └── index.html               # Flask frontend template
│
├── nasa.ipynb                   # NASA CMAPSS RUL prediction notebook
├── nasa_lstm_rul_model.h5       # Trained RUL model (legacy HDF5)
├── nasa_lstm_rul_model.keras    # Trained RUL model (Keras v3)
│
├── assets/                      # Screenshots and diagrams
├── requirement.txt              # Python dependencies
└── LICENSE
```

---

## ⚙️ System Architecture

```
┌─────────────────────┐
│      Arduino        │
│  MPU6050 | Flame    │
│  Sensor  | Temp     │
└────────┬────────────┘
         │  Serial (USB)
         ▼
┌─────────────────────┐
│   Python Backend    │
│  collect / monitor  │
└────────┬────────────┘
         │
         ▼  StandardScaler (scaler.save)
┌─────────────────────┐
│  LSTM Autoencoder   │  ← motor_anomaly_lstm.keras
│  Encoder → Decoder  │
└────────┬────────────┘
         │  MSE Reconstruction Error
         ▼
    ┌────┴────┐
    │Threshold│  ← compute_threshold.py
    └────┬────┘
         │
   ┌─────┴──────┐
   │            │
NORMAL       ANOMALY
   │            │
Green LED    Red LED (blink on critical)
   │            │
   └─────┬──────┘
         ▼
  Flask / Streamlit Dashboard
  (live graphs + anomaly timeline)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/thejuspk07/LSTM-Based-Multi-Sensor-Anomaly-Detection.git
cd LSTM-Based-Multi-Sensor-Anomaly-Detection
```

### 2. Install Dependencies

```bash
pip install -r requirement.txt
```

### 3. Flash Arduino Firmware

Open `arduino_led_control.ino` in the Arduino IDE and upload it to your board. Confirm the correct COM port is selected.

### 4. Verify Serial Connection

```bash
python arduino_stream_test.py
```

---

## ▶️ Running the System

### Option A — Collect Your Own Dataset

Run the motor under normal conditions and capture labelled data:

```bash
python collect_dataset.py
```

Then retrain the model using `g.ipynb` and recompute the threshold:

```bash
python compute_threshold.py
```

### Option B — Use Pre-trained Model

Skip collection and go straight to monitoring with the bundled `motor_anomaly_lstm.keras`:

```bash
# Terminal-based monitor
python motor_monitor.py

# OR lightweight detector
python real_time_detector.py
```

### Launch the Dashboard

**Flask** (web browser):
```bash
python app.py
# Open http://localhost:5000
```

**Streamlit** (live charts):
```bash
streamlit run dashboard.py
```

---

## 📈 Model Details

### Motor Anomaly — LSTM Autoencoder

| Parameter | Value |
|---|---|
| Architecture | Encoder → Latent → Decoder |
| Input features | AccX, AccY, AccZ, GyroX, GyroY, GyroZ, Temp, Flame |
| Sequence length | 30 timesteps |
| Anomaly metric | Mean Squared Error (reconstruction) |
| Threshold | Computed from 95th percentile of validation MSE |

The autoencoder is trained **only on normal data**. Any input that produces a reconstruction error above the threshold is flagged as an anomaly.

### NASA CMAPSS — LSTM RUL Predictor

The `nasa.ipynb` notebook trains a separate LSTM on the [NASA CMAPSS turbofan engine degradation dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) to predict **Remaining Useful Life (RUL)** in cycles.

| Parameter | Value |
|---|---|
| Dataset | NASA CMAPSS FD001 |
| Task | Regression (RUL prediction) |
| Architecture | Stacked LSTM |
| Output | Estimated remaining engine cycles |
| Saved model | `nasa_lstm_rul_model.keras` |

---

## 📊 Detection Workflow (Step-by-Step)

1. Arduino samples MPU6050 + flame + temperature sensors at fixed intervals
2. Readings are sent over USB serial as CSV rows
3. Python backend reads and buffers 30-timestep windows
4. Each window is normalised using the saved `StandardScaler`
5. The LSTM Autoencoder reconstructs the window
6. Mean Squared Error is computed between input and reconstruction
7. MSE is compared against the pre-computed threshold
8. If MSE > threshold → **ANOMALY**: red LED on, dashboard alert triggered
9. All events are logged and visualised in real time

---

## 🔮 Future Improvements

- [ ] MQTT / IoT cloud integration (AWS IoT, ThingsBoard)
- [ ] Email and SMS alerts on anomaly detection
- [ ] Edge AI deployment on ESP32 / Raspberry Pi
- [ ] Support for multiple simultaneous motors
- [ ] Automated retraining pipeline with new data
- [ ] Explainability layer — highlight which sensor triggered the anomaly
- [ ] Mobile app dashboard (Flutter / React Native)
- [ ] Predictive maintenance scheduling integration

---

## 🤝 Contributing

Contributions are welcome!

```bash
# 1. Fork the repo and create your branch
git checkout -b feature/your-feature-name

# 2. Commit your changes
git commit -m "feat: add your feature"

# 3. Push and open a Pull Request
git push origin feature/your-feature-name
```

Please open an issue first for major changes.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Jeffin Mathew** **Thejus P K**  
B.Tech AI & Data Science | Vimal Jyothi Engineering College (KTU)

If this project helped you, please consider giving it a ⭐ on GitHub!
