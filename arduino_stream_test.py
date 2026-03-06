import serial
import csv
import time

ser = serial.Serial('COM3', 9600)  # change COM port if needed
time.sleep(2)

with open('sensor_log.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['accel_x','accel_y','accel_z','temperature'])

    print("Collecting data... Press Ctrl+C to stop.")

    while True:
        line = ser.readline().decode().strip()
        values = line.split(',')

        if len(values) == 4:
            writer.writerow(values)
            print(values)