# ------------------------------------------------------------
# File: backend/api/app.py
# Project: AI Traffic Light Optimizer
# Role: API Integration by Shivam Paliwal
# Purpose: Serve vehicle count and signal decision using per-lane videos
# ------------------------------------------------------------

from flask import Flask, jsonify
from backend.video_input.video_common import get_live_video_frame
from backend.video_input.video_common import get_stream_managers
from backend.detection.vehicle_counter import detect_vehicles
from backend.logic.signal_controller import decide_signal
from ultralytics import YOLO
import sys, os
from backend.video_input.stream_manager import VideoStreamManager
import cv2
from backend.detection.model_utils import roads as ROAD_PATHS, vehicle_weights

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

# instantiate model
model = YOLO(MODEL_PATH)

import threading
import time

# ------------------------------------------------------------
# Initialize Flask App
# ------------------------------------------------------------
app = Flask(__name__)


# -----------------------------
# Approach (previously called 'lane') configuration and detector threads
# -----------------------------
APPROACH_ROADS = ["road1", "road2", "road3", "road4"]

# persistent controller for approach-level decisions
CONTROLLER = decide_signal(config=None, lanes=APPROACH_ROADS)

# shared structure storing latest detections per approach-road
APPROACH_RESULTS = {road: [] for road in APPROACH_ROADS}
APPROACH_LOCK = threading.Lock()


def approach_detector_thread(road, manager, model):
    """Thread loop: read frames from manager, run detect_vehicles, store latest boxes."""
    while True:
        try:
            frame = manager.next_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            boxes = detect_vehicles(frame, model) or []
            # compute count and total weight for this approach
            try:
                count = len(boxes)
                weight_sum = 0
                for (_x, _y, _w, _h, cls) in boxes:
                    weight_sum += int(vehicle_weights.get(cls, 2))
            except Exception:
                count = len(boxes)
                weight_sum = 0

            with APPROACH_LOCK:
                APPROACH_RESULTS[road] = {"count": count, "weight": weight_sum, "boxes": boxes}
        except Exception as e:
            print(f"[Detector thread] error on approach {road}: {e}")
            time.sleep(0.5)


# Start detector threads for approach-roads when their stream managers exist
STREAM_MANAGERS = get_stream_managers()
for road in APPROACH_ROADS:
    manager = STREAM_MANAGERS.get(road)
    if manager:
        t = threading.Thread(target=approach_detector_thread, args=(road, manager, model), daemon=True)
        t.start()
    else:
        print(f"[API] No stream manager for approach {road}; detector thread not started.")


# -----------------------------
# Per-road stream managers (each road video: roadA, roadB...)
# Paths in `ROAD_VIDEOS` are absolute or relative to project ROOT.
# -----------------------------
ROAD_STREAM_MANAGERS = {}
for road_name, rel_path in ROAD_PATHS.items():
    path = rel_path if os.path.isabs(rel_path) else os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        print(f"Warning: road video not found for {road_name} at {path}")
        continue
    try:
        ROAD_STREAM_MANAGERS[road_name] = VideoStreamManager(path)
    except Exception as e:
        print(f"[API] Road stream init failed for {road_name}: {e}")

# Road-level controller (separate bookkeeping)
ROAD_CONTROLLER = decide_signal(config=None, lanes=list(ROAD_PATHS.keys()))


# ------------------------------------------------------------
# API Endpoint 1: Get approach-wise vehicle count
# ------------------------------------------------------------
@app.route("/vehicle-count")
def vehicle_count():
    road_stats = {}
    try:
        with APPROACH_LOCK:
            for road in APPROACH_ROADS:
                data = APPROACH_RESULTS.get(road) or {"count": 0, "weight": 0, "boxes": []}
                road_stats[road] = {"count": int(data.get("count", 0)), "weight": int(data.get("weight", 0))}
        return jsonify(road_stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ------------------------------------------------------------
# API Endpoint 2: Get approach-wise signal decision (green road + time)
# ------------------------------------------------------------
@app.route("/signal-status")
def signal_status():
    road_stats = {}
    try:
        # build stats and feed controller with weight sums (more important than raw counts)
        controller_input = {}
        with APPROACH_LOCK:
            for road in APPROACH_ROADS:
                data = APPROACH_RESULTS.get(road) or {"count": 0, "weight": 0, "boxes": []}
                cnt = int(data.get("count", 0))
                w = int(data.get("weight", 0))
                road_stats[road] = {"count": cnt, "weight": w}
                # controller expects integer counts; use weight as proxy
                controller_input[road] = w
        print(f"[API] /signal-status | Final road_stats: {road_stats}")
    except Exception as e:
        print("Error in /signal-status:", e)
        return jsonify({"error": str(e)}), 500

    # use persistent controller; feed weights for decision
    served_road, green_time = CONTROLLER.run_once(controller_input)
    cycle = CONTROLLER._format_cycle(served_road, green_time, controller_input)
    return jsonify({"cycle": cycle, "stats": road_stats})



# ------------------------------------------------------------
# New: Per-road endpoints (roadA, roadB...)
# ------------------------------------------------------------
@app.route("/road-count")
def road_count():
    """Return vehicle counts per road video (roadA, roadB, ...)."""
    road_counts = {}
    try:
        for road, manager in ROAD_STREAM_MANAGERS.items():
            frame = None
            try:
                frame = manager.next_frame()
            except Exception as e:
                print(f"[API] road-count: failed to get frame for {road}: {e}")

            if frame is None:
                road_counts[road] = {"count": 0, "weight": 0}
                continue

            try:
                detections = detect_vehicles(frame, model) or []
            except Exception as e:
                print(f"[API] road-count: detection error for {road}: {e}")
                detections = []
            # compute weight sum
            wsum = 0
            for (_x, _y, _w, _h, cls) in detections:
                wsum += int(vehicle_weights.get(cls, 2))
            road_counts[road] = {"count": len(detections), "weight": wsum}
        return jsonify(road_counts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/road-status")
def road_status():
    """Return controller decision per road (which road to serve, green time)."""
    road_counts = {}
    try:
        for road, manager in ROAD_STREAM_MANAGERS.items():
            try:
                frame = manager.next_frame()
            except Exception as e:
                print(f"[API] road-status: failed to get frame for {road}: {e}")
                frame = None

            if frame is None:
                road_counts[road] = {"count": 0, "weight": 0}
                continue

            try:
                detections = detect_vehicles(frame, model) or []
            except Exception as e:
                print(f"[API] road-status: detection error for {road}: {e}")
                detections = []

            # compute weight sum
            wsum = 0
            for (_x, _y, _w, _h, cls) in detections:
                wsum += int(vehicle_weights.get(cls, 2))
            road_counts[road] = {"count": len(detections), "weight": wsum}

        # Use separate ROAD_CONTROLLER so per-road EWMA is independent
        # For road-level controller, use weight as input
        road_input = {r: int(d.get("weight", 0)) for r, d in road_counts.items()}
        served_road, green_time = ROAD_CONTROLLER.run_once(road_input)
        cycle = ROAD_CONTROLLER._format_cycle(served_road, green_time, road_input)
        return jsonify({"cycle": cycle, "stats": road_counts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ------------------------------------------------------------
# Run Flask App
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)