import cv2
import numpy as np
import time
import platform
from ultralytics import YOLO

# ---------------- SERIAL SETUP ----------------
try:
    import serial
    SERIAL_AVAILABLE = True
except:
    SERIAL_AVAILABLE = False

ser = None

def init_serial():
    global ser
    if not SERIAL_AVAILABLE:
        print("Serial library not available")
        return

    try:
        system = platform.system()

        if system == "Windows":
            port = "COM3"
        else:
            port = "/dev/ttyAMA0"

        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
        print(f"Serial connected on {port}")

    except Exception as e:
        print("Serial not connected:", e)
        ser = None

def send(cmd):
    msg = f"{cmd}\n"
    if ser:
        try:
            ser.write(msg.encode())
        except:
            pass
    print("CMD:", msg.strip())

# ---------------- CONFIG ----------------
WIDTH = 640
HEIGHT = 480

ROI_Y_START = int(HEIGHT * 0.3)

DIST_CONST = 550
DIST_THRESHOLD = 180     # earlier detection
EMERGENCY_DIST = 90      # immediate reverse

MIN_AREA = 900           # more sensitive motion

ALPHA = 0.9
steer_smoothed = 0

last_action = "FORWARD"

# ---------------- DISTANCE ----------------
def estimate_distance(h):
    if h <= 0:
        return 999
    return DIST_CONST / h

# ---------------- INIT ----------------
init_serial()

print("Loading YOLO...")
model = YOLO("yolov8n.pt")
print("Model loaded")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not opening")
    exit()

cap.set(3, WIDTH)
cap.set(4, HEIGHT)

time.sleep(2)

# ---------------- BACKGROUND ----------------
ret, prev_frame = cap.read()
if not ret:
    print("Cannot read camera")
    exit()

prev_frame = prev_frame[ROI_Y_START:HEIGHT]
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

print("ADAS Running (FAST MODE)")

# ---------------- LOOP ----------------
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    roi = frame[ROI_Y_START:HEIGHT]

    # ================= MOTION DETECTION =================
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    diff = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        motion_boxes.append((x, y + ROI_Y_START, w, h))

    prev_gray = gray

    # ================= YOLO (SKIPPED FRAMES FOR SPEED) =================
    frame_count += 1
    if frame_count % 2 == 0:
        results = model(frame, conf=0.45, imgsz=256, verbose=False)
    else:
        results = []

    human_boxes = []
    obstacle_boxes = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < 0.45:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            if h < 60:
                continue

            if cls == 0:
                human_boxes.append((x1, y1, w, h))
            elif cls in [1, 2, 3, 5, 7]:
                obstacle_boxes.append((x1, y1, w, h))

    # ================= MERGE =================
    all_boxes = []

    for (hx, hy, hw, hh) in human_boxes:
        for (mx, my, mw, mh) in motion_boxes:
            if (hx < mx + mw and hx + hw > mx and hy < my + mh and hy + hh > my):
                all_boxes.append(("HUMAN", (hx, hy, hw, hh)))
                break

    for (ox, oy, ow, oh) in obstacle_boxes:
        all_boxes.append(("OBSTACLE", (ox, oy, ow, oh)))

    for b in motion_boxes:
        all_boxes.append(("MOTION", b))

    # ================= STEERING =================
    steer = 0
    min_dist = 999

    left_blocked = False
    center_blocked = False
    right_blocked = False

    for label, (x, y, w, h) in all_boxes:

        dist = estimate_distance(h)

        if label == "HUMAN":
            dist *= 0.85

        cx = x + w // 2

        # Region classification
        if cx < WIDTH // 3:
            region = "LEFT"
        elif cx > 2 * WIDTH // 3:
            region = "RIGHT"
        else:
            region = "CENTER"

        if dist < DIST_THRESHOLD:
            if region == "LEFT":
                left_blocked = True
            elif region == "RIGHT":
                right_blocked = True
            else:
                center_blocked = True

        # Steering influence
        offset = (cx - WIDTH // 2) / (WIDTH // 2)
        weight = 2.5 / (dist + 1e-5)

        if label == "HUMAN":
            weight *= 1.5
        elif label == "MOTION":
            weight *= 2.0   # fast reaction

        steer += offset * weight

        min_dist = min(min_dist, dist)

        # Draw
        color = (0, 255, 0) if label == "HUMAN" else (255, 0, 0) if label == "OBSTACLE" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # ================= SMOOTH =================
    steer_smoothed = ALPHA * steer_smoothed + (1 - ALPHA) * steer
    steer_final = steer_smoothed * 0.6

    # ================= DECISION =================
    if min_dist < EMERGENCY_DIST:
        action = "BACK"

    elif min_dist < DIST_THRESHOLD:
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
        else:
            action = "FORWARD"

    else:
        if steer_final > 0.2:
            action = "RIGHT"
        elif steer_final < -0.2:
            action = "LEFT"
        else:
            action = "FORWARD"

    # ================= SEND =================
    if action != last_action:
        send(action)
        last_action = action

    # ================= DISPLAY =================
    cv2.putText(frame, f"STEER: {steer_final:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"ACTION: {action}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (0, ROI_Y_START), (WIDTH, ROI_Y_START), (255, 255, 0), 2)

    cv2.imshow("ADAS System (FAST)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()

if ser:
    ser.close()
