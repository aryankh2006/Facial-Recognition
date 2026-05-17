# Facial Recognition Drowsiness Detection

A real-time driver drowsiness detection project built with Python, OpenCV, and MediaPipe. The app uses a webcam feed to estimate eye closure, yawning, and head pose, then combines those signals into a single drowsiness score with an on-screen alert.

## Features

- Real-time webcam-based face tracking
- Eye Aspect Ratio (EAR) detection for closed eyes
- Mouth Aspect Ratio (MAR) detection for yawning
- Head pose estimation for nodding or tilting
- PERCLOS rolling-window fatigue measurement
- Weighted fusion score from 0 to 100
- Visual HUD with live metrics and drowsiness score
- Alert overlay when drowsiness remains high

## Project Files

```text
FacialRec.py            Main drowsiness detection script
face_landmarker.task    MediaPipe face landmark model
README.md               Project documentation
```

If `face_landmarker.task` is missing, the script attempts to download it automatically from Google's MediaPipe model storage.

## Requirements

- Python 3.9 or newer
- Webcam access
- Internet access for the first run if `face_landmarker.task` is not already present

Python packages:

```bash
pip3 install opencv-python "mediapipe>=0.10" numpy
```

## Setup

Clone the repository:

```bash
git clone https://github.com/aryankh2006/Facial-Recognition.git
cd Facial-Recognition
```

Optional but recommended: create a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip3 install opencv-python "mediapipe>=0.10" numpy
```

## Run

Start the detector:

```bash
python3 FacialRec.py
```

Press `q` in the camera window to quit.

## How It Works

The detector calculates several fatigue-related signals from facial landmarks:

- `EAR`: Eye Aspect Ratio. Lower values indicate closed or partially closed eyes.
- `MAR`: Mouth Aspect Ratio. Higher values indicate a possible yawn.
- `PERCLOS`: Percentage of recent frames where the eyes are closed.
- `Pitch` and `Roll`: Head angle estimates used to detect nodding or tilting.

These metrics are combined into a weighted drowsiness score:

```text
score = PERCLOS weight + yawn weight + head pose weight
```

An alert appears when the score stays above the alert threshold for a set duration.

## Tunable Settings

The main thresholds are near the top of `FacialRec.py`:

```python
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.6
PITCH_THRESHOLD = 15
ROLL_THRESHOLD = 20
ALERT_SCORE = 60
ALERT_DURATION = 2.0
PERCLOS_WINDOW = 90
```

You may need to adjust these values depending on lighting, camera position, face distance, and the user's natural facial features.

## Troubleshooting

### Webcam does not open

Make sure no other app is using the camera. On macOS, also check:

```text
System Settings > Privacy & Security > Camera
```

Allow camera access for your terminal or code editor.

### MediaPipe install fails

Upgrade `pip` first:

```bash
python3 -m pip install --upgrade pip
pip3 install opencv-python "mediapipe>=0.10" numpy
```

### Model download fails

Download `face_landmarker.task` manually and place it in the same folder as `FacialRec.py`.

Expected file name:

```text
face_landmarker.task
```

### GitHub push fails

Check the remote:

```bash
git remote -v
```

If SSH gives a public key error, use HTTPS:

```bash
git remote set-url origin https://github.com/aryankh2006/Facial-Recognition.git
git push origin main
```

## Disclaimer

This project is for learning and experimentation. It should not be used as a safety-critical driver monitoring system without proper validation, testing, and hardware calibration.
