import serial
import time
import numpy as np
from keras.models import load_model

ser = serial.Serial('COM7',9600)
time.sleep(2)

model = load_model("nasa_lstm_rul_model.keras")

while True:

    dummy_input = np.random.rand(1,30,17)

    prediction = model.predict(dummy_input)

    rul = prediction[0][0]

    print("Predicted RUL:",rul)

    if rul < 40:
        ser.write(b'1')
        print("ANOMALY DETECTED")

    else:
        ser.write(b'0')
        print("SYSTEM HEALTHY")

    time.sleep(3)