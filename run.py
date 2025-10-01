# ------------------------------------------------------------
# File: run.py
# Project: AI Traffic Light Optimizer
# Role: Integration Testing by Shivam Paliwal
# Purpose: Run detection → lane mapping → signal decision pipeline
# ------------------------------------------------------------

import cv2

# Import team modules
from backend.detection.vehicle_counter import detect_vehicles       # Member 1
from backend.lane_logic.lane_segmenter import map_to_lanes         # Member 3
from backend.logic.signal_controller import decide_signal          # Member 2

# Import real-time frame capture module
from backend.video_input.live_capture import get_live_frame        # live capture module

# ------------------------------------------------------------
# Step 1: Capture live frame from webcam
# ------------------------------------------------------------
try:
    frame = get_live_frame()
except Exception as e:
    raise RuntimeError(f"Failed to capture live frame: {e}")

# ------------------------------------------------------------
# Step 2: Define lane regions (x_min, x_max)
# ------------------------------------------------------------
# Update this if Member 3 modifies lane logic or makes it dynamic
lane_regions = {
    "lane1": (0, 200),
    "lane2": (201, 400),
    "lane3": (401, 600)
}

# ------------------------------------------------------------
# Step 3: Detect vehicles in frame
# ------------------------------------------------------------
# Member 1's function (do not modify)
vehicle_boxes = detect_vehicles(frame)

# ------------------------------------------------------------
# Step 4: Map vehicles to lanes
# ------------------------------------------------------------
lane_counts = map_to_lanes(vehicle_boxes, lane_regions)

# ------------------------------------------------------------
# Step 5: Decide which lane gets green signal
# ------------------------------------------------------------
signal = decide_signal(lane_counts)

# ------------------------------------------------------------
# Step 6: Output results
# ------------------------------------------------------------
print("Vehicle Counts per Lane:", lane_counts)
print("Signal Decision:", signal)