# ------------------------------------------------------------
# File: run.py
# Project: AI Traffic Light Optimizer
# Role: Integration Testing by Shivam Paliwal
# Purpose: Run detection → lane-wise counting → signal decision using video input
# ------------------------------------------------------------

from backend.video_input.video_common import VIDEO_PATHS, get_video_frame  # ✅ Centralized video logic
from backend.detection.vehicle_counter import detect_vehicles              # Member 1
from backend.logic.signal_controller import decide_signal                  # Member 2

# ------------------------------------------------------------
# Step 1: Detect vehicles per lane using video frames
# ------------------------------------------------------------
lane_counts = {}

for lane, path in VIDEO_PATHS.items():
    try:
        frame = get_video_frame(path)              # ✅ Load frame from video
        boxes = detect_vehicles(frame)             # ✅ Detect vehicles
        lane_counts[lane] = len(boxes)             # ✅ Count per lane
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