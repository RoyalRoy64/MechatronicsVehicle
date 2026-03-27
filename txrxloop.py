import serial
import time

# Use the correct port
# ser = serial.Serial('/dev/serial0', 115200, timeout=1)
ser = serial.Serial('/dev/ttyAMA0', 115200)


time.sleep(2)  # allow port to initialize

print("Starting UART loopback test...\n")

while True:
    msg = "HELLO_FROM_PI\n"
    
    # Send data
    ser.write(msg.encode())
    print("Sent:", msg.strip())

    time.sleep(0.5)

    # Read response (should be same due to loopback)
    if ser.in_waiting > 0:
        data = ser.readline().decode(errors='ignore').strip()
        print("Received:", data)
    else:
        print("No data received")

    print("------------------------")
    time.sleep(1)