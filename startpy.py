import serial
import time

# ser = serial.Serial('/dev/ttyUSB0', 115200)
# ser = serial.Serial('/dev/serial0', 115200)
ser = serial.Serial('/dev/ttyAMA0', 115200)

while True:
    print('start')
    ser.write(b'F\n')
    time.sleep(2)

    ser.write(b'R\n')
    time.sleep(1)

    ser.write(b'S\n')
    time.sleep(2)