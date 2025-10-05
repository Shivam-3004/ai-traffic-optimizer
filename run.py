# ------------------------------------------------------------
# File: run.py
# Project: AI Traffic Light Optimizer
# Role: Integration Testing by Shivam Paliwal
# Purpose: Run detection → lane-wise counting → signal decision using video input
# ------------------------------------------------------------

from backend.video_input.video_common import VIDEO_PATHS, get_sampled_frames  # ✅ Centralized video logic
from backend.detection.vehicle_counter import detect_vehicles              # Member 1
from backend.logic.signal_controller import decide_signal                  # Member 2

# ------------------------------------------------------------
# Step 1: Detect vehicles per lane using video frames
# ------------------------------------------------------------
lane_counts = {}
from ultralytics import YOLO
model = YOLO("backend/detection/models/best.pt")
for lane, path in VIDEO_PATHS.items():
    try:
        frames = get_sampled_frames(path, num_frames=5, stride=30)
        total_detected = 0
        for frame in frames:
            boxes = detect_vehicles(frame, model)
            total_detected += len(boxes)
        lane_counts[lane] = int(total_detected / max(1, len(frames)))
    except Exception as e:
        lane_counts[lane] = 0
        print(f"[Error] {lane}: {str(e)}")

# ------------------------------------------------------------
# Step 2: Decide which lane gets green signal
# ------------------------------------------------------------
signal = decide_signal(lane_counts)

# ------------------------------------------------------------
# Step 3: Output results
# ------------------------------------------------------------
print("✅ Vehicle Counts per Lane:", lane_counts)
print("🚦 Signal Decision:", signal)