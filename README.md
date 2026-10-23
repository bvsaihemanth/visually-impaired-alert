# Blind Assistance System Using YOLOv8 and Bayesian Risk Assessment

A real-time vision assistance prototype that detects objects from a live
camera stream, estimates their approximate distance, evaluates risk with
a Bayesian Network, and provides visual and spoken alerts.

## Architecture

![Blind Assistance System
Architecture](assets/blind_assistance_architecture.png)

## Features

-   **Live camera input:** Reads video from an IP Webcam stream using
    OpenCV.
-   **Object detection:** Uses a YOLOv8 model to identify objects and
    confidence scores.
-   **Approximate distance estimation:** Estimates distance from
    detected bounding-box height.
-   **Bayesian risk assessment:** Combines object type, confidence, and
    proximity to estimate risk.
-   **Visual feedback:** Draws bounding boxes, labels, estimated
    distance, and FPS on the video.
-   **Voice alerts:** Uses Google Text-to-Speech (gTTS) and `playsound`
    to announce nearby objects.
-   **Performance summary:** Reports processed frames, detections,
    alerts, estimated false alerts, alert precision, and average FPS.

## How It Works

1.  OpenCV receives frames from the camera stream.
2.  YOLOv8 detects objects in each frame.
3.  The system estimates distance from bounding-box height and
    categorizes object type, detection confidence, and proximity.
4.  A Bayesian Network estimates the probability of high risk.
5.  Distance thresholds and the high-risk probability determine whether
    an alert is triggered.
6.  The application displays annotated video and plays periodic spoken
    alerts.

## Requirements

-   Python 3.10 or a compatible Python version
-   A working camera stream, such as an IP Webcam app
-   YOLOv8 weights compatible with Ultralytics
-   The Python packages listed below

## Installation

Create and activate a virtual environment:

``` bash
python -m venv .venv
```

Windows PowerShell:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

``` bash
pip install ultralytics opencv-python numpy gTTS playsound pgmpy
```

If audio playback fails, check the `playsound` version and your
operating system's audio support.

## Configuration

Update these values near the top of your Python script:

``` python
camera_url = "http://YOUR_CAMERA_IP:4747/video"
model_path = r"C:\path\to\your\yolov8s-oiv7.pt"
```

-   `camera_url`: Set this to the video URL shown by your IP Webcam
    application. Your computer and phone generally need to be reachable
    on the same network.
-   `model_path`: Set this to the actual local path of your YOLO weights
    file.
-   `LARGE_OBJECT_THRESHOLD` and `SMALL_OBJECT_THRESHOLD`: Configure the
    distance thresholds used by the alert rules.
-   `SPEECH_INTERVAL`: Configure how often an active voice alert
    repeats.

Do not commit private files, credentials, personal data, datasets, or
large model weights unless you intentionally want to publish them.

## Run

Save the Python code as a script, for example `blind_assistance.py`,
then run:

``` bash
python blind_assistance.py
```

Press **`q`** in the video window to exit.

## Important Limitations

-   **Distance is approximate.** The current formula uses bounding-box
    height and a fixed constant; it is not calibrated depth estimation.
    Accuracy can vary with object size, camera position, and viewing
    angle.
-   **Risk thresholds are hand-designed.** The Bayesian Network
    probabilities and alert thresholds should be validated with
    representative data.
-   **Alert precision is a custom estimate.** The current `false_alerts`
    counter is based on a distance-rule check, not manually verified
    ground truth. The printed alert precision should not be treated as a
    standard evaluated precision metric.
-   **Safety:** This is a prototype, not a certified mobility aid. It
    must not be relied on as the sole means of navigation or personal
    safety.

## Project Structure

``` text
your-project/
├── blind_assistance.py
├── README.md
└── assets/
    └── blind_assistance_architecture.png
```

Adjust the example script filename to match your repository.

## License

Add a license file if you intend to distribute this project. No license
is specified here.
