from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort  # SORT tracker

# Load YOLO model
model = YOLO("yolo11n.pt")

# Load video
cap = cv2.VideoCapture("5330829-hd_1920_1080_30fps.mp4")  # Or 0 for webcam

# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Counters
left_to_right = set()
right_to_left = set()
previous_positions = {}

# Optional: Resize for speed (keep None to use original size)
resize_width = 1280  # Change to None to keep original resolution
resize_height = None  # Auto-calculated if width is set

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    if resize_width:
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        if resize_height is None:
            resize_height = int(resize_width / aspect_ratio)
        frame = cv2.resize(frame, (resize_width, resize_height))

    height, width = frame.shape[:2]
    line_x = width // 2  # Vertical center line

    # YOLO inference
    results = model(frame, verbose=False)
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            if label.lower() == "person" and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, conf])

    if len(detections) > 0:
        tracked = tracker.update(np.array(detections))
    else:
        tracked = tracker.update()

    for track in tracked:
        x1, y1, x2, y2, track_id = map(int, track)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw tracking box & ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Direction detection
        if track_id in previous_positions:
            prev_cx = previous_positions[track_id]
            if prev_cx < line_x and cx >= line_x:
                left_to_right.add(track_id)
            elif prev_cx > line_x and cx <= line_x:
                right_to_left.add(track_id)

        previous_positions[track_id] = cx

    # Draw vertical center line
    cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 2)

    # Display counts
    cv2.putText(frame, f"Left -> Right: {len(left_to_right)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"Right -> Left: {len(right_to_left)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.putText(frame, f"Total: {len(left_to_right) + len(right_to_left)}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("People Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()