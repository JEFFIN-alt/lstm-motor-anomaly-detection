import serial
import csv
import time

port = "COM7"
baud = 9600

ser = serial.Serial(port, baud)
time.sleep(2)

file = open("sensor_dataset.csv", "w", newline="")
writer = csv.writer(file)

writer.writerow(["ax","ay","az","gx","gy","gz","temp","flame"])

print("Collecting data...")

start = time.time()

while time.time() - start < 300:   # 5 minutes
    line = ser.readline().decode().strip()

    if line:
        values = line.split(",")
        print(values)
        writer.writerow(values)

file.close()
ser.close()

print("Dataset saved as sensor_dataset.csv")