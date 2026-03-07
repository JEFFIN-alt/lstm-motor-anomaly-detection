"""
compute_threshold.py
Runs the saved LSTM autoencoder on the NORMAL training data
and computes the correct anomaly threshold as:
    mean(MSE) + 3 * std(MSE)   (99.7% of normal stays below this)

Also prints a recommended range so you can decide.
"""
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

SEQUENCE_LEN  = 30
FEATURE_COLS  = ["ax","ay","az","gx","gy","gz","temp","flame"]

print("Loading model and scaler...")
model  = load_model("motor_anomaly_lstm.keras")
scaler = joblib.load("scaler.save")

print("Loading normal training data...")
df = pd.read_csv("motor_normal_labeled.csv")

# Keep only label==0 (normal) rows that have all 8 features
df = df[df["label"] == 0][FEATURE_COLS].dropna()

# Drop rows that couldn't be parsed (the 'MPU6050 connection failed' header row etc.)
df = df.apply(pd.to_numeric, errors='coerce').dropna()

print(f"Normal samples: {len(df)}")

# Scale using the same scaler used during training
# Pass a DataFrame with column names to match what the scaler was fitted with
scaled = scaler.transform(df[FEATURE_COLS])

# Build sequences
X_list = []
for i in range(len(scaled) - SEQUENCE_LEN + 1):
    X_list.append(scaled[i : i + SEQUENCE_LEN])

X = np.array(X_list, dtype=np.float32)
print(f"Sequences to test: {len(X)}")

# Run inference in one batch (faster than loop)
print("Running inference on all normal sequences...")
reconstructions = model.predict(X, batch_size=64, verbose=1)

# Per-sequence MSE
mse_per_seq = np.mean(np.square(X - reconstructions), axis=(1, 2))

mean_mse = np.mean(mse_per_seq)
std_mse  = np.std(mse_per_seq)
p95      = np.percentile(mse_per_seq, 95)
p99      = np.percentile(mse_per_seq, 99)
p999     = np.percentile(mse_per_seq, 99.9)

print("\n========== THRESHOLD ANALYSIS ==========")
print(f"  Mean MSE (normal data)    : {mean_mse:.6f}")
print(f"  Std  MSE (normal data)    : {std_mse:.6f}")
print()
print(f"  95th percentile           : {p95:.6f}   ← loose  (catches mild anomalies)")
print(f"  99th percentile           : {p99:.6f}   ← balanced (recommended)")
print(f"  99.9th percentile         : {p999:.6f}   ← strict  (catches strong anomalies)")
print(f"  Mean + 3*Std              : {mean_mse + 3*std_mse:.6f}   ← classic rule")
print()
print(f"  Your OLD threshold        : 0.0628")
print(f"  Threshold you just set    : 0.1628")
print()
print("=== RECOMMENDED ===")
recommended = p99
print(f"  Use THRESHOLD = {recommended:.6f}   (99th percentile of normal data)")
print("========================================")
