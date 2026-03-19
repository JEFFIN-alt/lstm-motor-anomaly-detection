import serial
import time

COM_PORT = "COM7"
BAUD_RATE = 9600

print("Connecting to Arduino...")

ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)

time.sleep(2)

print("Reading sensor data...")

while True:

    try:

        line = ser.readline().decode().strip()

        if line:
            print(line)

    except KeyboardInterrupt:

        print("Stopped")
        break