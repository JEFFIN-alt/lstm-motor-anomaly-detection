import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

SEQUENCE_LEN = 30

FEATURES = [
    "ax","ay","az",
    "gx","gy","gz",
    "temp",
    "flame"
]

print("Loading model and scaler...")

model = load_model("motor_anomaly_lstm.keras")
scaler = joblib.load("scaler.save")

print("Loading dataset...")

df = pd.read_csv("motor_dataset.csv")

data = scaler.transform(df[FEATURES])

sequences = []

for i in range(len(data) - SEQUENCE_LEN):

    sequences.append(data[i:i+SEQUENCE_LEN])

X = np.array(sequences)

print("Computing reconstruction error...")

pred = model.predict(X)

errors = np.mean((X - pred)**2, axis=(1,2))

threshold = np.mean(errors) + 3*np.std(errors)

print("\nRecommended LSTM Threshold:")

print(threshold)