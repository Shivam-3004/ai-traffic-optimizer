# ------------------------------------------------------------
# File: backend/api/app.py
# Project: AI Traffic Light Optimizer
# Role: API Integration by Shivam Paliwal
# Purpose: Serve real-time vehicle count and signal decision
# ------------------------------------------------------------

from flask import Flask, jsonify

# Import real-time frame capture module
from backend.video_input.live_capture import get_live_frame         # live capture module

# Import team modules
from backend.detection.vehicle_counter import detect_vehicles       # Member 1
from backend.lane_logic.lane_segmenter import map_to_lanes         # Member 3
from backend.logic.signal_controller import decide_signal          # Member 2

# Initialize Flask app
app = Flask(__name__)

# Define lane regions (x_min, x_max)
# Update this if Member 3 modifies lane logic or makes it dynamic
lane_regions = {
    "lane1": (0, 200),
    "lane2": (201, 400),
    "lane3": (401, 600)
}

# ------------------------------------------------------------
# API Endpoint 1: Get lane-wise vehicle count
# ------------------------------------------------------------
@app.route("/vehicle-count")
def vehicle_count():
    try:
        # Capture live frame from webcam
        frame = get_live_frame()
    except Exception as e:
        return jsonify({"error": f"Failed to capture frame: {str(e)}"}), 500

    # Step 1: Detect vehicles (Member 1)
    vehicle_boxes = detect_vehicles(frame)

    # Step 2: Map vehicles to lanes (Member 3)
    lane_counts = map_to_lanes(vehicle_boxes, lane_regions)

    # Return lane-wise count as JSON
    return jsonify(lane_counts)

# ------------------------------------------------------------
# API Endpoint 2: Get signal decision (green lane + time)
# ------------------------------------------------------------
@app.route("/signal-status")
def signal_status():
    try:
        # Capture live frame from webcam
        frame = get_live_frame()
    except Exception as e:
        return jsonify({"error": f"Failed to capture frame: {str(e)}"}), 500

    # Step 1: Detect vehicles (Member 1)
    vehicle_boxes = detect_vehicles(frame)

    # Step 2: Map vehicles to lanes (Member 3)
    lane_counts = map_to_lanes(vehicle_boxes, lane_regions)

    # Step 3: Decide signal (Member 2)
    signal = decide_signal(lane_counts)

    # Return signal decision as JSON
    return jsonify(signal)

# ------------------------------------------------------------
# Run the Flask app
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)