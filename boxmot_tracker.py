import cv2
import numpy as np
from ultralytics import YOLO
# Use the correct import for your BoxMOT version
from boxmot import ByteTrack

# ============================================================
# CONFIGURATION
# ============================================================
VIDEO_PATH = "video.mp4"
OUTPUT_PATH = "boxmot_output.mp4"
YOLO_MODEL = "yolov8n.pt"
TARGET_CLASS_ID = 0   # 'person' class ID in COCO dataset
CONFIDENCE_THRESHOLD = 0.35
SHOW_LIVE_WINDOW = True

# ============================================================
# INITIALIZATION
# ============================================================
print("Loading YOLO model...")
model = YOLO(YOLO_MODEL)
# If you have a compatible GPU, uncomment the next line for a massive speed boost
# model.to('cuda')

print("Initializing ByteTrack tracker...")
tracker = ByteTrack()
# Optional: Reset the tracker ID for a new run
tracker.reset()

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for the output writer
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_count = 0

print("Starting tracking loop...")
# ============================================================
# MAIN PROCESSING LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # --- 1. Run YOLO Detection on the current frame ---
    # results = model(frame, verbose=False)[0]  # Standard way
    results = model(frame, verbose=False)[0]

    # --- 2. Format detections for BoxMOT ---
    # BoxMOT expects a NumPy array in the format:
    # [x1, y1, x2, y2, confidence, class_id]
    dets_list = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == TARGET_CLASS_ID and conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dets_list.append([x1, y1, x2, y2, conf, cls_id])
    
    # Convert the list to a NumPy array as required by the tracker
    dets = np.array(dets_list) if dets_list else np.empty((0, 6))

    # --- 3. Update the tracker with the detections ---
    # The tracker uses the detections and the original frame to output active tracks.
    tracks = tracker.update(dets, frame)

    # --- 4. Draw the tracking results on the frame ---
    # 'tracks' is a NumPy array. Each row is a track with format:
    # [x1, y1, x2, y2, track_id, conf, class_id, ...]
    if tracks.size > 0:
        for track in tracks:
            # Extract only the first 7 values which we need for drawing
            x1, y1, x2, y2, track_id, conf, cls_id = track[:7]
            # Convert to integers for drawing
            x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw track ID and confidence
            label = f'ID: {track_id} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Optional: Display live window
    if SHOW_LIVE_WINDOW:
        resized_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("YOLO + BoxMOT Tracking", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Write the frame to the output video
    out.write(frame)

    # Optional: Print progress every 100 frames
    if frame_count % 100 == 0:
        print(f"Processed frame {frame_count}...")

# ============================================================
# CLEANUP
# ============================================================
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Tracking complete. Output saved to {OUTPUT_PATH}")
