# ------------------------------------------------------------
# File: backend/api/app.py
# Project: AI Traffic Light Optimizer
# ------------------------------------------------------------

from flask import Flask, jsonify
from backend.video_input.video_common import get_stream_managers
from backend.detection.vehicle_counter import detect_vehicles
from backend.logic.signal_controller import decide_signal
from backend.detection.model_utils import vehicle_weights
from ultralytics import YOLO
import sys, os, time

# ------------------------------------------------------------
# Path Setup
# ------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# ------------------------------------------------------------
# Load YOLO Model Once
# ------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(ROOT, "backend", "detection", "models", "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found at: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# ------------------------------------------------------------
# Flask App Setup
# ------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------
# Road Configuration
# ------------------------------------------------------------
APPROACH_ROADS = ["road1", "road2", "road3", "road4"]
ROAD_NAMES = ["Road 1", "Road 2", "Road 3", "Road 4"]
STREAM_MANAGERS = get_stream_managers()

# Persistent signal controller
CONTROLLER = decide_signal(config=None, lanes=APPROACH_ROADS)

# Track the current green phase
CURRENT_PHASE = {"lane": None, "end_time": 0.0}

# ------------------------------------------------------------
# API 1: Vehicle Count
# ------------------------------------------------------------
@app.route("/vehicle-count")
def vehicle_count():
    """Return live vehicle count keyed by Road 1..4."""
    road_stats = {}
    try:
        for road, road_name in zip(APPROACH_ROADS, ROAD_NAMES):
            manager = STREAM_MANAGERS.get(road)
            if not manager:
                road_stats[road_name] = {"count": 0, "weight": 0}
                continue

            frame = manager.next_frame()
            if frame is None:
                road_stats[road_name] = {"count": 0, "weight": 0}
                continue

            detections = detect_vehicles(frame, model) or []
            weight_sum = sum(int(vehicle_weights.get(cls, 2)) for (_x, _y, _w, _h, cls) in detections)
            road_stats[road_name] = {"count": len(detections), "weight": int(weight_sum)}

        return jsonify(road_stats)
    except Exception as e:
        print(f"[ERROR] vehicle_count: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------
# API 2: Signal Status
# ------------------------------------------------------------
@app.route("/signal-status")
def signal_status():
    """Return current signal cycle keyed by Road 1..4. Red time = 0."""
    try:
        controller_input = {}
        now = time.time()

        # Get counts from each road
        for road, road_name in zip(APPROACH_ROADS, ROAD_NAMES):
            manager = STREAM_MANAGERS.get(road)
            if not manager:
                controller_input[road] = 0
                continue

            frame = manager.next_frame()
            if frame is None:
                controller_input[road] = 0
                continue

            detections = detect_vehicles(frame, model) or []
            weight_sum = sum(int(vehicle_weights.get(cls, 2)) for (_x, _y, _w, _h, cls) in detections)
            controller_input[road] = weight_sum

        # -------------------------------
        # Decide green lane respecting CURRENT_PHASE
        # -------------------------------
        if CURRENT_PHASE["lane"] is None or now >= CURRENT_PHASE["end_time"]:
            served_road = CONTROLLER.select_lane(controller_input)

            # Prevent same lane immediately repeating
            if served_road == CURRENT_PHASE["lane"]:
                CONTROLLER.service_order = "round_robin"
                served_road = CONTROLLER.select_lane(controller_input)
                CONTROLLER.service_order = "auto"

            green_time = CONTROLLER.calculate_green_time(CONTROLLER.smoothed_counts.get(served_road, 0))
            CURRENT_PHASE["lane"] = served_road
            CURRENT_PHASE["end_time"] = now + green_time
        else:
            served_road = CURRENT_PHASE["lane"]
            green_time = CURRENT_PHASE["end_time"] - now

        # Build cycle dictionary keyed by Road X
        cycle = {}
        for road, road_name in zip(APPROACH_ROADS, ROAD_NAMES):
            if road == served_road:
                status = "Green"
                time_val = int(round(green_time))
            else:
                status = "Red"
                time_val = 0
            cycle[road_name] = {"status": status, "time": time_val}

        return jsonify({"cycle": cycle})

    except Exception as e:
        print("Error in /signal-status:", e)
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------
# Run Flask Server
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🚦 AI Traffic Light Optimizer API Running on http://127.0.0.1:5000")
    print("✅ Routes available:")
    print("   → /vehicle-count  (for live counts)")
    print("   → /signal-status  (for signal decision)")
    app.run(debug=True, port=5000)
