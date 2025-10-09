"""
API module for AI Traffic Light Optimizer (commented)

Responsibilities:
 - Load a single YOLO model instance for inference
 - Initialize per-road video stream managers
 - Expose two Flask endpoints:
     1) /vehicle-count -> returns per-road vehicle counts and weights
     2) /signal-status  -> returns current signal cycle (which road is green and remaining time)

Important design notes:
 - Stream managers are expected to run their own background threads and return
   the latest frame via `next_frame()` (non-blocking call).
 - A single persistent `CONTROLLER` (decide_signal) object is used so that
   smoothing, starvation timers and logging are retained across requests.
 - `CURRENT_PHASE` holds the active green lane and its expiration timestamp.
 - `PHASE_LOCK` prevents race conditions where multiple simultaneous requests
   might try to advance the phase.

Behavioral contract (API outputs):
 - /vehicle-count returns a JSON mapping of human-friendly keys ``"Road 1"``.. to
   an object {"count": int, "weight": int} where count = number of detections
   and weight = sum of vehicle_weights lookup.
 - /signal-status returns {"cycle": {"Road 1": {"status":"Red|Green","time":int}, ...}}
   Only the green road includes a positive time; red roads report time=0.

"""

from flask import Flask, jsonify
from flask_cors import CORS
from backend.video_input.video_common import get_stream_managers
from backend.detection.vehicle_counter import detect_vehicles
from backend.logic.signal_controller import decide_signal
from backend.detection.model_utils import vehicle_weights
from ultralytics import YOLO
import sys, os, time
import threading
import atexit

# Process-level start info for debugging restarts
START_TIME = time.time()
PID = os.getpid()

# Path setup: allow running this module directly in various CWDs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Load model once at process startup. This avoids reloading the weights per-request.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(ROOT, "backend", "detection", "models", "best.pt")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found at: {MODEL_PATH}")
# The global `model` should be reused for inference across requests
model = YOLO(MODEL_PATH)

# Flask app
app = Flask(__name__)
CORS(app)

# Road configuration: internal ids and friendly names. Keep these in sync with other modules/UI.
APPROACH_ROADS = ["road1", "road2", "road3", "road4"]
ROAD_NAMES = ["Road 1", "Road 2", "Road 3", "Road 4"]

# Stream managers provide non-blocking access to the latest frame for each road.
STREAM_MANAGERS = get_stream_managers()

# Persistent controller instance (handles EWMA smoothing, starvation, logging)
CONTROLLER = decide_signal(config=None, lanes=APPROACH_ROADS)

# CURRENT_PHASE tracks which road currently has the green and when it ends.
# Use PHASE_LOCK to avoid races when multiple HTTP clients call /signal-status concurrently.
CURRENT_PHASE = {"lane": None, "end_time": 0.0}
PHASE_LOCK = threading.Lock()


@app.route("/vehicle-count")
def vehicle_count():
    """Return live per-road counts and weights.

    Returns (JSON): {
        "Road 1": {"count": int, "weight": int},
        "Road 2": {...},
        ...
    }

    Notes:
    - If a stream manager or frame is unavailable, count and weight are 0 for that road.
    - The function is intentionally defensive: it catches exceptions and returns HTTP 500
      with an error message if something unexpected happens.
    """
    road_stats = {}
    try:
        for road, road_name in zip(APPROACH_ROADS, ROAD_NAMES):
            manager = STREAM_MANAGERS.get(road)
            if not manager:
                # Stream manager missing -> return zeros for that road
                road_stats[road_name] = {"count": 0, "weight": 0}
                continue

            # next_frame() should be non-blocking and return the latest frame (or None)
            frame = manager.next_frame()
            if frame is None:
                # No frame available right now
                road_stats[road_name] = {"count": 0, "weight": 0}
                continue

            # Run detection on the latest frame
            detections = detect_vehicles(frame, model) or []
            # vehicle_weights is a dict mapping class name -> integer weight
            weight_sum = sum(int(vehicle_weights.get(cls, 2)) for (_x, _y, _w, _h, cls) in detections)
            road_stats[road_name] = {"count": len(detections), "weight": int(weight_sum)}

        return jsonify(road_stats)
    except Exception as e:
        # Return error details for debugging (in production, be more conservative)
        print(f"[ERROR] vehicle_count: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/signal-status")
def signal_status():
    """Return the current signal cycle

    Behavior:
    - Build controller input by computing weighted vehicle counts for each road.
    - Use a phase lock to ensure only one request can advance the phase at a time.
    - If the current green phase has expired (or none set), call CONTROLLER.run_once()
      which performs smoothing, selects a lane, computes green_time, updates internal state,
      and logs the decision.
    - If a green phase is active, do NOT re-run selection; instead return the same green lane
      and remaining time. This ensures the green is "locked" for its full duration.

    Response schema (JSON): {"cycle": {"Road 1": {"status": "Red|Green", "time": int}, ...}}
    Only the green road will have a positive time; red roads show time=0.
    """
    try:
        now = time.time()
        controller_input = {}

        # Build the controller input (weighted counts). This is the same information
        # that run_once() expects; by delegating to run_once() we avoid duplicating
        # selection / smoothing logic.
        for road in APPROACH_ROADS:
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

        with PHASE_LOCK:
            # If no active phase or current green expired -> let controller decide
            if CURRENT_PHASE["lane"] is None or now >= CURRENT_PHASE["end_time"]:
                # Delegate to controller: it will update smoothing and last_served internally.
                served_road, green_time = CONTROLLER.run_once(controller_input)
                # Store the chosen lane and when it should expire
                CURRENT_PHASE["lane"] = served_road
                CURRENT_PHASE["end_time"] = now + float(green_time)
            else:
                # Phase still active; return remaining time for the active lane
                served_road = CURRENT_PHASE["lane"]
                green_time = max(0.0, CURRENT_PHASE["end_time"] - now)

        # Build human-friendly cycle output keyed by ROAD_NAMES
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


# If run directly, start the Flask dev server. In production use a WSGI server.
@app.route("/meta")
def meta():
    """Return process metadata so clients can detect server restarts (PID/start_time)."""
    return jsonify({"pid": PID, "start_time": int(START_TIME)})


if __name__ == "__main__":
    print("🚦 AI Traffic Light Optimizer API Running on http://127.0.0.1:5000")
    print("✅ Routes available:")
    print("   → /vehicle-count  (for live counts)")
    print("   → /signal-status  (for signal decision)")
    # Run without the debugger to ensure no auto-reload/watchers will restart the process
    # when the controller writes logs. Enable threaded to handle concurrent requests.
    app.run(debug=False, port=5000, use_reloader=False, threaded=True)