"""
Enhanced Driver Drowsiness Detection
-------------------------------------
Detection methods:
  1. EAR      – Eye Aspect Ratio (eye closure)
  2. MAR      – Mouth Aspect Ratio (yawning)
  3. Head pose – pitch/roll from MediaPipe face matrix
  4. Gyro     – MPU6050 angular velocity (head movement speed)
  5. FSR      – Pressure sensor (driver seated gate)
  6. Fusion   – Weighted combination → single drowsiness score

Dependencies:
    pip3 install opencv-python "mediapipe>=0.10" numpy pyserial
"""

import cv2
import time
import math
import numpy as np
from collections import deque
import serial
import serial.tools.list_ports

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker, RunningMode

# ──────────────────────────────────────────────
# Arduino / Serial config
# ──────────────────────────────────────────────
ARDUINO_PORT  = None   # None = auto-detect, or set e.g. '/dev/tty.usbmodem1101'
ARDUINO_BAUD  = 9600
FSR_PIN       = 4      # ESP32S3 analog pin for FSR
FSR_THRESHOLD = 500    # raw ADC value — above = driver seated

# ──────────────────────────────────────────────
# Drowsiness thresholds
# ──────────────────────────────────────────────
EAR_THRESHOLD      = 0.22   # eye aspect ratio below this = closed
MAR_THRESHOLD      = 0.6    # mouth aspect ratio above this = yawning
PITCH_THRESHOLD    = 15     # degrees forward head nod
ROLL_THRESHOLD     = 20     # degrees sideways tilt
YAWN_MIN_DURATION  = 1.5    # seconds mouth open to count as yawn
ALERT_SCORE        = 60     # fusion score to trigger alert
ALERT_DURATION     = 2.0    # seconds score must stay above threshold
PERCLOS_WINDOW     = 90     # rolling frame window (~3s at 30fps)

# Gyro thresholds (rad/s) — sudden head movement = drowsy jerk
GYRO_JERK_THRESHOLD = 1.5   # rad/s — fast head snap indicates microsleep recovery
GYRO_WINDOW         = 30    # frames to track gyro activity

# Fusion weights (must sum to 1.0)
W_PERCLOS   = 0.35
W_YAWN      = 0.20
W_HEAD_POSE = 0.25
W_GYRO      = 0.20

# ──────────────────────────────────────────────
# MediaPipe landmark indices
# ──────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]
NOSE_TIP  = 1
CHIN      = 152
LEFT_EYE_L  = 226
RIGHT_EYE_R = 446
LEFT_MOUTH  = 57
RIGHT_MOUTH = 287

# ──────────────────────────────────────────────
# Serial helpers
# ──────────────────────────────────────────────
def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        if any(k in desc for k in ("arduino", "usbmodem", "usbserial", "ch340", "cp210", "esp")):
            return p.device
    return ports[0].device if ports else None


def connect_arduino():
    port = ARDUINO_PORT or find_arduino_port()
    if port is None:
        print("No Arduino found — running without sensors.")
        return None
    try:
        ser = serial.Serial(port, ARDUINO_BAUD, timeout=0.1)
        time.sleep(2)
        print(f"Arduino connected on {port}")
        return ser
    except Exception as e:
        print(f"Could not connect to Arduino ({e}) — running without sensors.")
        return None


def parse_serial(line):
    """
    Parse CSV line from Arduino: fsr,ax,ay,az,gx,gy,gz
    Returns (fsr, ax, ay, az, gx, gy, gz) or None on error.
    """
    try:
        parts = line.strip().split(",")
        if len(parts) == 7:
            return tuple(float(p) for p in parts)
    except Exception:
        pass
    return None

# ──────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_indices]
    A = euclidean(pts[2], pts[6])
    B = euclidean(pts[3], pts[7])
    C = euclidean(pts[0], pts[1])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def get_head_angles(landmarks, w, h):
    model_points = np.array([
        [0.0,    0.0,    0.0   ],
        [0.0,   -330.0, -65.0 ],
        [-225.0, 170.0, -135.0],
        [225.0,  170.0, -135.0],
        [-150.0,-150.0, -125.0],
        [150.0, -150.0, -125.0],
    ], dtype=np.float64)
    def lm(idx): return (landmarks[idx].x * w, landmarks[idx].y * h)
    image_points = np.array([
        lm(NOSE_TIP), lm(CHIN),
        lm(LEFT_EYE_L), lm(RIGHT_EYE_R),
        lm(LEFT_MOUTH), lm(RIGHT_MOUTH),
    ], dtype=np.float64)
    focal_length = w
    cam_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    success, rvec, _ = cv2.solvePnP(
        model_points, image_points, cam_matrix, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll  = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll  = 0.0
    return pitch, roll

# ──────────────────────────────────────────────
# Overlay helpers
# ──────────────────────────────────────────────
BAR_X, BAR_Y, BAR_W, BAR_H = 20, 20, 200, 22


def draw_bar(frame, y, label, value, max_val, color_low, color_high, threshold=None):
    """Generic horizontal bar."""
    filled = int(BAR_W * min(value, max_val) / max_val)
    pct    = value / max_val * 100
    color  = color_high if value / max_val > 0.6 else color_low
    cv2.rectangle(frame, (BAR_X, y), (BAR_X + BAR_W, y + BAR_H), (50, 50, 50), -1)
    cv2.rectangle(frame, (BAR_X, y), (BAR_X + filled, y + BAR_H), color, -1)
    cv2.rectangle(frame, (BAR_X, y), (BAR_X + BAR_W, y + BAR_H), (200, 200, 200), 1)
    if threshold is not None:
        tx = BAR_X + int(BAR_W * threshold / max_val)
        cv2.line(frame, (tx, y - 2), (tx, y + BAR_H + 2), (255, 255, 0), 2)
    cv2.putText(frame, f"{label}: {pct:.0f}%",
                (BAR_X, y + BAR_H + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)


def draw_hud(frame, ear, mar, pitch, roll, perclos,
             fsr_raw, gyro_mag, driver_present, arduino_connected):
    """Bottom-left metrics panel."""
    h = frame.shape[0]
    seat_str = "Seated" if driver_present else "NOT SEATED"
    gyro_str = f"{gyro_mag:.2f} rad/s"
    lines = [
        f"EAR:      {ear:.3f}",
        f"MAR:      {mar:.3f}",
        f"Pitch:    {pitch:.1f}deg",
        f"Roll:     {roll:.1f}deg",
        f"PERCLOS:  {perclos*100:.1f}%",
        f"Gyro mag: {gyro_str}" if arduino_connected else "Gyro: no sensor",
        f"Pressure: {fsr_raw}" if arduino_connected else "Pressure: no sensor",
        f"Seat:     {seat_str}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (10, h - 20 - (len(lines) - 1 - i) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)


def draw_alert(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.putText(frame, "DROWSINESS ALERT!",
                (40, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, "Please pull over and rest.",
                (60, frame.shape[0] // 2 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    import urllib.request, os
    MODEL_PATH = "face_landmarker.task"
    MODEL_URL  = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    )
    if not os.path.exists(MODEL_PATH):
        print("Downloading face_landmarker.task (~30 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")

    arduino = connect_arduino()

    options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    face_landmarker = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # ── State ──────────────────────────────────
    perclos_buffer  = deque(maxlen=PERCLOS_WINDOW)
    gyro_buffer     = deque(maxlen=GYRO_WINDOW)   # magnitude of gyro each frame
    ear_consec      = 0
    yawn_start      = None
    yawn_count      = 0
    alert_start     = None
    alerting        = False
    driver_present  = True
    fsr_raw         = 0
    gyro_mag        = 0.0   # current gyro magnitude (rad/s)
    ax = ay = az    = 0.0
    gx = gy = gz    = 0.0

    print("Drowsiness detection running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        h, w = frame.shape[:2]

        # ── Read from Arduino (CSV: fsr,ax,ay,az,gx,gy,gz) ──
        if arduino and arduino.in_waiting > 0:
            try:
                line = arduino.readline().decode('utf-8')
                parsed = parse_serial(line)
                if parsed:
                    fsr_raw, ax, ay, az, gx, gy, gz = parsed
                    driver_present = fsr_raw > FSR_THRESHOLD
                    gyro_mag = math.sqrt(gx**2 + gy**2 + gz**2)
            except Exception:
                pass

        # Track gyro magnitude over time
        gyro_buffer.append(gyro_mag)

        # ── MediaPipe face detection ──────────
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        timestamp_ms = int(time.time() * 1000)
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        ear = mar = pitch = roll = 0.0
        face_detected = False

        if result.face_landmarks:
            face_detected = True
            lm = result.face_landmarks[0]

            # EAR
            left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            eye_closed = ear < EAR_THRESHOLD
            perclos_buffer.append(1 if eye_closed else 0)
            ear_consec = ear_consec + 1 if eye_closed else 0

            # MAR / Yawn
            mar = mouth_aspect_ratio(lm, MOUTH, w, h)
            if mar > MAR_THRESHOLD:
                if yawn_start is None:
                    yawn_start = time.time()
                elif time.time() - yawn_start >= YAWN_MIN_DURATION:
                    yawn_count += 1
                    yawn_start = None
            else:
                yawn_start = None

            # Head pose
            if result.facial_transformation_matrixes:
                mat = np.array(result.facial_transformation_matrixes[0].data).reshape(4, 4)
                r = mat[:3, :3]
                sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
                pitch = math.degrees(math.atan2(-r[2, 0], sy))
                roll  = math.degrees(math.atan2(r[2, 1], r[2, 2])) if sy > 1e-6 else 0.0
            else:
                pitch, roll = get_head_angles(lm, w, h)

            # Draw landmark dots
            for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
                cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 2, (0, 255, 255), -1)

        # ── Scores ────────────────────────────
        perclos = np.mean(perclos_buffer) if perclos_buffer else 0.0

        pitch_score = min(abs(pitch) / PITCH_THRESHOLD, 1.0)
        roll_score  = min(abs(roll)  / ROLL_THRESHOLD,  1.0)
        head_score  = max(pitch_score, roll_score)

        yawn_score  = min(yawn_count / 3.0, 1.0)

        # Gyro score: high average magnitude = jerky head movements (microsleep recovery)
        avg_gyro    = np.mean(gyro_buffer) if gyro_buffer else 0.0
        gyro_score  = min(avg_gyro / GYRO_JERK_THRESHOLD, 1.0) if arduino else 0.0

        # Fusion
        if face_detected:
            fusion = (
                W_PERCLOS   * perclos    +
                W_YAWN      * yawn_score +
                W_HEAD_POSE * head_score +
                W_GYRO      * gyro_score
            ) * 100
        else:
            fusion = 0.0

        # ── Alert ─────────────────────────────
        if fusion >= ALERT_SCORE and driver_present:
            if alert_start is None:
                alert_start = time.time()
            alerting = (time.time() - alert_start) >= ALERT_DURATION
        else:
            alert_start = None
            alerting = False

        # ── Draw ──────────────────────────────
        if alerting:
            draw_alert(frame)

        # Bars (stacked)
        draw_bar(frame, BAR_Y,
                 "Drowsiness", fusion, 100,
                 (0, 255, 0), (0, 0, 255),
                 threshold=ALERT_SCORE)

        if arduino:
            draw_bar(frame, BAR_Y + BAR_H + 30,
                     "Pressure", fsr_raw, 4095,
                     (0, 0, 255), (0, 255, 0),
                     threshold=FSR_THRESHOLD)
            draw_bar(frame, BAR_Y + (BAR_H + 30) * 2,
                     "Gyro activity", avg_gyro, GYRO_JERK_THRESHOLD,
                     (0, 255, 0), (0, 165, 255))

        draw_hud(frame, ear, mar, pitch, roll, perclos,
                 fsr_raw, gyro_mag, driver_present,
                 arduino_connected=(arduino is not None))

        # Status tags top-right
        tags = []
        if not face_detected:
            tags.append(("No face detected", (0, 165, 255)))
        if face_detected and ear < EAR_THRESHOLD:
            tags.append(("Eyes closed", (0, 0, 255)))
        if face_detected and mar > MAR_THRESHOLD:
            tags.append(("Yawning", (0, 165, 255)))
        if face_detected and abs(pitch) > PITCH_THRESHOLD:
            tags.append(("Head nodding", (0, 165, 255)))
        if arduino and gyro_mag > GYRO_JERK_THRESHOLD:
            tags.append(("Head jerk detected", (0, 165, 255)))
        if arduino and not driver_present:
            tags.append(("NOT SEATED — alerts paused", (0, 0, 255)))

        for i, (tag, col) in enumerate(tags):
            cv2.putText(frame, tag, (w - 260, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

        cv2.putText(frame, f"Yawns: {yawn_count}", (w - 120, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    if arduino:
        arduino.close()


if __name__ == "__main__":
    main()