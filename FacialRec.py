"""
Enhanced Driver Drowsiness Detection
-------------------------------------
Detection methods:
  1. EAR  – Eye Aspect Ratio (eye closure)
  2. MAR  – Mouth Aspect Ratio (yawning)
  3. Head pose – pitch/roll nodding via facial landmarks
  4. Fusion score – weighted combination → single drowsiness level

Dependencies:
    pip install opencv-python mediapipe scipy numpy
"""

import cv2
import time
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
import mediapipe as mp

# ──────────────────────────────────────────────
# Tunable thresholds
# ──────────────────────────────────────────────
EAR_THRESHOLD       = 0.22   # below → eye considered closed
MAR_THRESHOLD       = 0.6    # above → mouth considered open (yawn)
PITCH_THRESHOLD     = 15     # degrees forward head tilt (nodding)
ROLL_THRESHOLD      = 20     # degrees sideways tilt

EAR_CONSEC_FRAMES   = 3      # frames EAR must stay low before counting
YAWN_MIN_DURATION   = 1.5    # seconds mouth must stay open to count as yawn

ALERT_SCORE         = 60     # fusion score (0-100) that triggers alert
ALERT_DURATION      = 2.0    # seconds score must stay above threshold

# Rolling window for PERCLOS (last N frames)
PERCLOS_WINDOW      = 90     # ~3 s at 30 fps

# Fusion weights (must sum to 1.0)
W_PERCLOS   = 0.45
W_YAWN      = 0.25
W_HEAD_POSE = 0.30

# ──────────────────────────────────────────────
# MediaPipe landmark indices
# ──────────────────────────────────────────────
# Left eye
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
# Mouth (outer)
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]
# Head pose reference points
NOSE_TIP      = 1
CHIN          = 152
LEFT_EYE_L    = 226
RIGHT_EYE_R   = 446
LEFT_MOUTH    = 57
RIGHT_MOUTH   = 287


# ──────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    # Vertical distances
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    # Horizontal distance
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]
    # Vertical distances (top-bottom pairs)
    A = dist.euclidean(pts[2], pts[6])
    B = dist.euclidean(pts[3], pts[7])
    # Horizontal distance
    C = dist.euclidean(pts[0], pts[1])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def get_head_angles(landmarks, w, h):
    """
    Estimate pitch (nod) and roll (tilt) using solvePnP.
    Returns (pitch_deg, roll_deg).
    """
    model_points = np.array([
        [0.0,    0.0,    0.0   ],   # nose tip
        [0.0,   -330.0, -65.0 ],   # chin
        [-225.0, 170.0, -135.0],   # left eye left corner
        [225.0,  170.0, -135.0],   # right eye right corner
        [-150.0,-150.0, -125.0],   # left mouth corner
        [150.0, -150.0, -125.0],   # right mouth corner
    ], dtype=np.float64)

    def lm(idx):
        return (landmarks[idx].x * w, landmarks[idx].y * h)

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
    dist_coeffs = np.zeros((4, 1))

    success, rvec, _ = cv2.solvePnP(
        model_points, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    # Decompose into Euler angles
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll  = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll  = 0.0
    return pitch, roll


# ──────────────────────────────────────────────
# Overlay helpers
# ──────────────────────────────────────────────
BAR_X, BAR_Y, BAR_W, BAR_H = 20, 20, 200, 24

def draw_score_bar(frame, score):
    """Draw a colour-coded drowsiness score bar."""
    filled = int(BAR_W * score / 100)
    color = (0, 255, 0) if score < 40 else (0, 165, 255) if score < ALERT_SCORE else (0, 0, 255)
    cv2.rectangle(frame, (BAR_X, BAR_Y), (BAR_X + BAR_W, BAR_Y + BAR_H), (50, 50, 50), -1)
    cv2.rectangle(frame, (BAR_X, BAR_Y), (BAR_X + filled, BAR_Y + BAR_H), color, -1)
    cv2.rectangle(frame, (BAR_X, BAR_Y), (BAR_X + BAR_W, BAR_Y + BAR_H), (200, 200, 200), 1)
    cv2.putText(frame, f"Drowsiness: {score:.0f}%", (BAR_X, BAR_Y + BAR_H + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def draw_metrics(frame, ear, mar, pitch, roll, perclos):
    """Small HUD in bottom-left."""
    h = frame.shape[0]
    lines = [
        f"EAR:     {ear:.3f}",
        f"MAR:     {mar:.3f}",
        f"Pitch:   {pitch:.1f}deg",
        f"Roll:    {roll:.1f}deg",
        f"PERCLOS: {perclos*100:.1f}%",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, h - 20 - (len(lines) - 1 - i) * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def draw_alert(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.putText(frame, "⚠  DROWSINESS ALERT!", (40, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 3)
    cv2.putText(frame, "Please pull over and rest.", (60, frame.shape[0] // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # ── State ──────────────────────────────────
    perclos_buffer   = deque(maxlen=PERCLOS_WINDOW)   # 1 = closed, 0 = open
    ear_consec       = 0

    yawn_start       = None
    yawn_count       = 0

    alert_start      = None  # when score crossed ALERT_SCORE
    alerting         = False

    print("Drowsiness detection running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Default metric values (no face detected)
        ear = mar = 0.0
        pitch = roll = 0.0
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            lm = results.multi_face_landmarks[0].landmark

            # ── EAR ────────────────────────────
            left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            eye_closed = ear < EAR_THRESHOLD
            perclos_buffer.append(1 if eye_closed else 0)

            if eye_closed:
                ear_consec += 1
            else:
                ear_consec = 0

            # ── MAR / Yawn ──────────────────────
            mar = mouth_aspect_ratio(lm, MOUTH, w, h)

            if mar > MAR_THRESHOLD:
                if yawn_start is None:
                    yawn_start = time.time()
                elif time.time() - yawn_start >= YAWN_MIN_DURATION:
                    yawn_count += 1
                    yawn_start = None          # reset for next yawn
            else:
                yawn_start = None

            # ── Head pose ───────────────────────
            pitch, roll = get_head_angles(lm, w, h)

            # Draw landmark dots for eyes and mouth
            for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
                x_px = int(lm[idx].x * w)
                y_px = int(lm[idx].y * h)
                cv2.circle(frame, (x_px, y_px), 2, (0, 255, 255), -1)

        # ── PERCLOS ────────────────────────────
        perclos = np.mean(perclos_buffer) if perclos_buffer else 0.0

        # ── Head pose score (0-1) ───────────────
        pitch_score = min(abs(pitch) / PITCH_THRESHOLD, 1.0)
        roll_score  = min(abs(roll)  / ROLL_THRESHOLD,  1.0)
        head_score  = max(pitch_score, roll_score)

        # ── Yawn score (decays slowly) ──────────
        # Each yawn contributes ~33 points; decays after 60 s
        yawn_score = min(yawn_count / 3.0, 1.0)

        # ── Fusion score (0-100) ───────────────
        if face_detected:
            fusion = (
                W_PERCLOS   * perclos   +
                W_YAWN      * yawn_score +
                W_HEAD_POSE * head_score
            ) * 100
        else:
            fusion = 0.0   # can't assess without a face

        # ── Alert logic ────────────────────────
        if fusion >= ALERT_SCORE:
            if alert_start is None:
                alert_start = time.time()
            alerting = (time.time() - alert_start) >= ALERT_DURATION
        else:
            alert_start = None
            alerting = False

        # ── Draw overlay ───────────────────────
        if alerting:
            draw_alert(frame)

        draw_score_bar(frame, fusion)
        draw_metrics(frame, ear, mar, pitch, roll, perclos)

        # Status tags
        if not face_detected:
            cv2.putText(frame, "No face detected", (w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        if ear < EAR_THRESHOLD and face_detected:
            cv2.putText(frame, "Eyes closed", (w - 160, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        if mar > MAR_THRESHOLD and face_detected:
            cv2.putText(frame, "Yawning", (w - 130, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
        if abs(pitch) > PITCH_THRESHOLD and face_detected:
            cv2.putText(frame, "Head nodding", (w - 170, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

        cv2.putText(frame, f"Yawns: {yawn_count}", (w - 120, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()