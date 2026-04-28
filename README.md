
markdown
# OpenCV Trackers – YOLO + ByteTrack Multi‑Object Tracking

This repository contains a **real‑time multi‑object tracking system** built with [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics) and [BoxMOT](https://github.com/mikel-brostrom/boxmot)’s **ByteTrack** algorithm. The project was developed as part of an OpenCV image processing course.

## Features

- **Multi‑object tracking** – track every person (or any other class) in the video simultaneously.
- **Unique IDs** – each tracked object keeps a persistent ID across frames.
- **Robust tracking** – uses ByteTrack with Kalman filtering and IoU association to handle occlusions and fast motion.
- **Easy customisation** – change target class, confidence threshold, or swap to other BoxMOT trackers (BoT‑SORT, DeepOCSORT, StrongSORT).
- **Lightweight** – runs efficiently on CPU; GPU acceleration available via CUDA.

## Demo

| Input Video | Output Video (tracked) |
|-------------|------------------------|
| (your video) | Annotated boxes with IDs |

## Requirements

- Python 3.9 – 3.12
- OpenCV (`opencv-python`)
- NumPy
- Ultralytics YOLO
- BoxMOT

All dependencies can be installed via `pip` or `conda`.

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/ArturoR1986/opencv_trackers.git
   cd opencv_trackers
Create a virtual environment (optional but recommended)

bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
Install required packages

bash
pip install ultralytics boxmot opencv-python numpy
For GPU acceleration, install PyTorch with CUDA first – see pytorch.org.

Usage
Place your input video file in the repository folder and name it video.mp4 (or change VIDEO_PATH in the script).

Run the tracker:

bash
python boxmot_tracker.py
The output video will be saved as boxmot_output.mp4 (or as set in OUTPUT_PATH).

Live Camera Feed (RTSP)
To use a security camera RTSP stream, replace VIDEO_PATH with your camera URL:

python
VIDEO_PATH = "rtsp://username:password@ip_address:port/stream"
Configuration
Edit the variables at the top of boxmot_tracker.py:

Variable	Description	Default
VIDEO_PATH	Path to input video or RTSP stream	video.mp4
OUTPUT_PATH	Path for output video	boxmot_output.mp4
YOLO_MODEL	YOLO model file (.pt)	yolov8n.pt
TARGET_CLASS_ID	COCO class ID to track (0 = person)	0
CONFIDENCE_THRESHOLD	Minimum detection confidence (0–1)	0.35
SHOW_LIVE_WINDOW	Show live preview window (True/False)	True
Changing the Tracker
You can easily switch to other BoxMOT trackers:

python
from boxmot import BotSort, DeepOCSORT, StrongSort

# For appearance-based re-identification (long occlusions)
tracker = BotSort(reid_weights='osnet_x0_25_msmt17.pt')
Output Format
The script draws a green bounding box around each tracked object with its unique ID and the detection confidence. The annotated video is saved to OUTPUT_PATH.

Troubleshooting
ModuleNotFoundError: No module named 'boxmot'
Run pip install boxmot.

Low FPS on CPU
Reduce image size via YOLO’s imgsz parameter or use a smaller model like yolov8n. Also try running inference on every 2nd or 3rd frame.

OpenCV window not showing
Set SHOW_LIVE_WINDOW = False if running in a headless environment.

Future Improvements
Add zone crossing / line counting

Integrate with Home Assistant for alerts

Export tracking data to CSV/JSON

Support for multiple camera streams

Credits
Ultralytics YOLO

BoxMOT

OpenCV community

License
This project is open source and available under the MIT License.

Author: ArturoR1986
Course: OpenCV Image Processing

text

---

You can copy this entire block into a file named `README.md` and upload it to your repository's root folder. If you want a shorter version (e.g., for the GitHub "About" section), let me know and I'll condense it.
