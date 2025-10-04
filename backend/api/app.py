# ------------------------------------------------------------
# File: backend/api/app.py
# Project: AI Traffic Light Optimizer
# Role: API Integration by Shivam Paliwal
# Purpose: Serve vehicle count and signal decision using per-lane videos
# ------------------------------------------------------------

from flask import Flask, jsonify
from backend.video_input.video_common import get_live_video_frame
from backend.detection.vehicle_counter import detect_vehicles
from backend.logic.signal_controller import decide_signal
from ultralytics import YOLO
import sys, os

# ------------------------------------------------------------
# Path Setup
# ------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# ------------------------------------------------------------
# Load YOLO Model Once
# ------------------------------------------------------------
model = YOLO("backend/detection/models/best.pt")  # adjust path if needed

# ------------------------------------------------------------
# Initialize Flask App
# ------------------------------------------------------------
app = Flask(__name__)

# ✅ Fixed lane order
LANES = ["lane1", "lane2", "lane3", "lane4"]

# ------------------------------------------------------------
# API Endpoint 1: Get lane-wise vehicle count
# ------------------------------------------------------------
@app.route("/vehicle-count")
def vehicle_count():
    lane_counts = {}
    try:
        for lane in LANES:
            frame = get_live_video_frame(lane)
            boxes = detect_vehicles(frame, model) or []
            lane_counts[lane] = len(boxes)
        return jsonify(lane_counts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------
# API Endpoint 2: Get signal decision (green lane + time)
# ------------------------------------------------------------
@app.route("/signal-status")
def signal_status():
    lane_counts = {}
    try:
        for lane in LANES:
            frame = get_live_video_frame(lane)
            print(f"[API] /signal-status | Lane {lane}: 1 live frame")
            try:
                boxes = detect_vehicles(frame, model) or []
            except Exception as e:
                print(f"[API] Detection error on {lane}: {e}")
                boxes = []
            lane_counts[lane] = len(boxes)
        print(f"[API] /signal-status | Final lane_counts: {lane_counts}")
    except Exception as e:
        print("Error in /signal-status:", e)
        return jsonify({"error": str(e)}), 500

    controller = decide_signal(config=None, lanes=LANES)
    served_lane, green_time = controller.run_once(lane_counts)
    cycle = controller._format_cycle(served_lane, green_time, lane_counts)
    return jsonify(cycle)

# ------------------------------------------------------------
# Run Flask App
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)