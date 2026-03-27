import cv2
import numpy as np
import time
from ultralytics import YOLO
import serial

ser = serial.Serial('/dev/ttyAMA0', 115200)

# ---------------- CONFIG ----------------
WIDTH = 640
HEIGHT = 480
DIST_CONST = 550
MIN_AREA = 1200   # motion sensitivity

# ---------------- DISTANCE ----------------
def estimate_distance(h):
    if h <= 0:
        return 999
    return DIST_CONST / h

# ---------------- LOAD YOLO ----------------
model = YOLO("yolov8n.pt")  # fast model

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Camera not opening")
    exit()

cap.set(3, WIDTH)
cap.set(4, HEIGHT)
time.sleep(2)

# ---------------- BACKGROUND ----------------
ret, prev_frame = cap.read()
if not ret:
    print(" Cannot read camera")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

print(" YOLO HYBRID ADAS STARTED")

# ---------------- SERIAL ----------------

# direction = "F"
# speed = 150

# # Create the message dynamically
# msg = f"{direction}:{speed}\n"  # e.g. "F:150\n"

# # Convert to bytes and send
# ser.write(msg.encode())

# print("Sent:", msg.strip())
def send(cmd):
    # print(f"COMMAND: {cmd}")
    msg = f"{cmd[0]}\n"
    ser.write(msg.encode())
    print(msg)

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # ================= MOTION DETECTION =================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    frame_diff = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        (x, y, w, h) = cv2.boundingRect(cnt)
        motion_boxes.append((x, y, w, h))

    prev_gray = gray

    # ================= YOLO DETECTION =================
    results = model(frame, conf=0.5,  verbose=False) #imgsz=320,

    human_boxes = []
    obstacle_boxes = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w = x2 - x1
            h = y2 - y1

            if h < 80:
                continue

            # 👤 HUMAN
            if cls == 0:
                human_boxes.append((x1, y1, w, h))

            # 🚗 VEHICLE / OBSTACLE
            elif cls in [1, 2, 3, 5, 7]:
                obstacle_boxes.append((x1, y1, w, h))

    # ================= MERGE =================
    all_boxes = []

    # Humans (must be moving)
    for (hx, hy, hw, hh) in human_boxes:
        for (mx, my, mw, mh) in motion_boxes:
            if (hx < mx + mw and hx + hw > mx and
                hy < my + mh and hy + hh > my):

                all_boxes.append(("HUMAN", (hx, hy, hw, hh)))
                break

    # Vehicles (always obstacle)
    for (ox, oy, ow, oh) in obstacle_boxes:
        all_boxes.append(("OBSTACLE", (ox, oy, ow, oh)))

    # Unknown motion
    for b in motion_boxes:
        all_boxes.append(("MOTION", b))

    # ================= DECISION =================
    min_dist = 999
    left_blocked = False
    center_blocked = False
    right_blocked = False

    for label, (x, y, w, h) in all_boxes:

        dist = estimate_distance(h)

        #  give higher priority to humans
        if label == "HUMAN":
            dist *= 0.8

        center_x = x + w // 2

        if dist < 120:
            if center_x < WIDTH // 3:
                left_blocked = True
            elif center_x > 2 * WIDTH // 3:
                right_blocked = True
            else:
                center_blocked = True

        if dist < min_dist:
            min_dist = dist

        #  Colors
        if label == "HUMAN":
            color = (0, 255, 0)      # Green
        elif label == "OBSTACLE":
            color = (255, 0, 0)      # Blue
        else:
            color = (0, 0, 255)      # Red

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {dist:.1f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ================= NAVIGATION =================
    action = "FORWARD"

    if min_dist < 120:
        if center_blocked:
            if not left_blocked:
                action = "LEFT"
            elif not right_blocked:
                action = "RIGHT"
            else:
                action = "BACK"
        elif left_blocked:
            action = "RIGHT"
        elif right_blocked:
            action = "LEFT"

    send(action)

    # ================= DISPLAY =================
    cv2.putText(frame, f"ACTION: {action}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("YOLO HYBRID ADAS", frame)

    if action != "FORWARD":
        print(f" OBSTACLE DETECTED → {action}")
    else:
        print(" CLEAR PATH")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()