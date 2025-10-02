# ------------------------------------------------------------
# File: backend/api/app.py
# Project: AI Traffic Light Optimizer
# Role: API Integration by Shivam Paliwal
# Purpose: Serve vehicle count and signal decision using per-lane videos
# ------------------------------------------------------------

from flask import Flask, jsonify
from backend.video_input.video_common import VIDEO_PATHS, get_video_frame  # ✅ Centralized video logic
from backend.detection.vehicle_counter import detect_vehicles
from backend.logic.signal_controller import decide_signal

# Initialize Flask app
app = Flask(__name__)

# ------------------------------------------------------------
# API Endpoint 1: Get lane-wise vehicle count
# ------------------------------------------------------------
@app.route("/vehicle-count")
def vehicle_count():
    lane_counts = {}
    try:
        for lane, path in VIDEO_PATHS.items():
            frame = get_video_frame(path)
            boxes = detect_vehicles(frame)
            lane_counts[lane] = len(boxes)
    except Exception as e:
        return jsonify({"error": f"Failed to process video frames: {str(e)}"}), 500

    return jsonify(lane_counts)

# ------------------------------------------------------------
# API Endpoint 2: Get signal decision (green lane + time)
# ------------------------------------------------------------
@app.route("/signal-status")
def signal_status():
    lane_counts = {}
    try:
        for lane, path in VIDEO_PATHS.items():
            frame = get_video_frame(path)
            boxes = detect_vehicles(frame)
            lane_counts[lane] = len(boxes)
    except Exception as e:
        return jsonify({"error": f"Failed to process video frames: {str(e)}"}), 500

    signal = decide_signal(lane_counts)
    return jsonify(signal)

# ------------------------------------------------------------
# Run the Flask app
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)