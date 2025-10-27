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
from backend.detection.vehicle_counter import detect_vehicles, draw_detections_on_frame, create_trackers, update_trackers
from backend.logic.signal_controller import decide_signal
from backend.detection.model_utils import vehicle_weights
from ultralytics import YOLO
import cv2
from flask import Response, request
from backend.video_input import video_common
from backend.video_input.stream_manager import VideoStreamManager
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

# Toggle: if True, the background detection worker runs and caches boxes for overlays.
# If False (default), detection is performed on-demand when endpoints are called or
# when a client connects to the raw stream. This prevents the camera being opened
# and inference running constantly when no clients are connected.
BACKGROUND_DET = False

# Persistent controller instance (handles EWMA smoothing, starvation, logging)
CONTROLLER = decide_signal(config=None, lanes=APPROACH_ROADS)

# CURRENT_PHASE tracks which road currently has the green and when it ends.
# Use PHASE_LOCK to avoid races when multiple HTTP clients call /signal-status concurrently.
CURRENT_PHASE = {"lane": None, "end_time": 0.0}
PHASE_LOCK = threading.Lock()

# Note: annotated-frame workers and /detected-* endpoints have been removed.

# Lightweight detection cache for live overlay (run for live camera road(s))
_DETECTION_CACHE = {}  # road -> {boxes: [...], width: int, height: int, ts: float}
_DETECTION_LOCK = threading.Lock()
# per-road detection threads & running flags
_DETECT_THREADS = {}  # road -> Thread
_DETECT_RUNNING = {}  # road -> bool
_DETECT_THREAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
DETECT_INTERVAL = 1.0  # seconds between detections for overlay (lower frequency -> lower CPU)
INFERENCE_RESIZE_WIDTH = 480  # resize frames for inference to reduce CPU (keep <= 640)
STREAM_FPS = 15
STREAM_INTERVAL = 1.0 / float(STREAM_FPS)

# trackers per road for smoother boxes between detections
_TRACKERS = {}  # road -> list of (tracker, class_name)
_TRACKERS_LOCK = threading.Lock()
_TRACKER_THREADS = {}
_TRACKER_RUNNING = {}
_TRACKER_THREAD_LOCK = threading.Lock()
TRACKER_FPS = 20
TRACKER_INTERVAL = 1.0 / float(TRACKER_FPS)

def _tracker_worker(road, manager):
    """Update trackers at a higher frequency to produce smooth boxes between detections."""
    try:
        while _TRACKER_RUNNING.get(road, False):
            frame = manager.next_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                with _TRACKERS_LOCK:
                    trackers = _TRACKERS.get(road)
                if not trackers:
                    time.sleep(TRACKER_INTERVAL)
                    continue
                boxes, new_trackers = update_trackers(frame, trackers)
                with _TRACKERS_LOCK:
                    _TRACKERS[road] = new_trackers
                # write boxes into detection cache so clients can poll them
                try:
                    h, w = frame.shape[:2]
                except Exception:
                    w = None; h = None
                with _DETECTION_LOCK:
                    _DETECTION_CACHE[road] = {"boxes": [{"x": float(b[0]), "y": float(b[1]), "w": float(b[2]), "h": float(b[3]), "class": b[4] if len(b) > 4 else ''} for b in boxes], "width": w, "height": h, "ts": time.time()}
            except Exception:
                pass
            time.sleep(TRACKER_INTERVAL)
    except Exception:
        return

def start_tracker_for_road(road):
    manager = STREAM_MANAGERS.get(road)
    if manager is None:
        return
    with _TRACKER_THREAD_LOCK:
        if _TRACKER_RUNNING.get(road):
            return
        _TRACKER_RUNNING[road] = True
        th = threading.Thread(target=_tracker_worker, args=(road, manager), daemon=True)
        _TRACKER_THREADS[road] = th
        th.start()

def stop_tracker_for_road(road):
    with _TRACKER_THREAD_LOCK:
        _TRACKER_RUNNING[road] = False
        th = _TRACKER_THREADS.get(road)
        if th and th.is_alive():
            try:
                th.join(timeout=0.5)
            except Exception:
                pass
        _TRACKER_THREADS.pop(road, None)


def _detection_worker(road, manager):
    """Run detection periodically on frames from manager and cache box coords.

    This worker resizes frames for inference to reduce CPU, scales boxes back to
    original frame coordinates, converts xywh(center) -> xy(top-left), and
    stores them in _DETECTION_CACHE for other endpoints (raw_stream, detection_boxes).
    """
    try:
        while _DETECT_RUNNING.get(road, False):
            frame = manager.next_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            # perform resized inference to save CPU
            try:
                with _INFERENCE_LOCK:
                    # resize if wider than target
                    h, w = frame.shape[:2]
                    scale = 1.0
                    resized = frame
                    if w > INFERENCE_RESIZE_WIDTH:
                        scale = INFERENCE_RESIZE_WIDTH / float(w)
                        resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    dets = detect_vehicles(resized, model) or []
            except Exception:
                dets = []

            # scale detections back to original size and convert centers -> top-left
            boxes = []
            try:
                h, w = frame.shape[:2]
            except Exception:
                w = None; h = None

            for d in dets:
                try:
                    if len(d) >= 5:
                        x_c, y_c, bw, bh, cls = float(d[0]), float(d[1]), float(d[2]), float(d[3]), str(d[4])
                        # scale up
                        if scale != 1.0:
                            x_c = x_c / scale
                            y_c = y_c / scale
                            bw = bw / scale
                            bh = bh / scale
                        # convert center->top-left
                        x_t = x_c - (bw / 2.0)
                        y_t = y_c - (bh / 2.0)
                        boxes.append({"x": float(x_t), "y": float(y_t), "w": float(bw), "h": float(bh), "class": cls})
                except Exception:
                    continue

            # create lightweight trackers for smoother updates between detections
            try:
                trackers = create_trackers(frame, [(b['x'], b['y'], b['w'], b['h'], b.get('class', '')) for b in boxes])
                with _TRACKERS_LOCK:
                    _TRACKERS[road] = trackers
            except Exception:
                pass

            with _DETECTION_LOCK:
                _DETECTION_CACHE[road] = {"boxes": boxes, "width": w, "height": h, "ts": time.time()}

            time.sleep(DETECT_INTERVAL)
    except Exception:
        # ensure any unexpected worker exception doesn't kill the process
        return


def start_detection_for_road(road):
    """Start a background detection worker for a specific road (idempotent)."""
    manager = STREAM_MANAGERS.get(road)
    if manager is None:
        return
    with _DETECT_THREAD_LOCK:
        if _DETECT_RUNNING.get(road):
            return
        _DETECT_RUNNING[road] = True
        th = threading.Thread(target=_detection_worker, args=(road, manager), daemon=True)
        _DETECT_THREADS[road] = th
        th.start()
        # also start lightweight tracker updater for smoother boxes
        try:
            start_tracker_for_road(road)
        except Exception:
            pass


def stop_detection_for_road(road):
    """Stop the background detection worker for a specific road."""
    with _DETECT_THREAD_LOCK:
        _DETECT_RUNNING[road] = False
        th = _DETECT_THREADS.get(road)
        if th and th.is_alive():
            try:
                th.join(timeout=1.0)
            except Exception:
                pass
        _DETECT_THREADS.pop(road, None)
        # remove cached detection for road to avoid stale boxes
        with _DETECTION_LOCK:
            _DETECTION_CACHE.pop(road, None)
        # remove trackers for this road as well and stop tracker thread
        try:
            stop_tracker_for_road(road)
        except Exception:
            pass
        try:
            with _TRACKERS_LOCK:
                _TRACKERS.pop(road, None)
        except Exception:
            pass


def stop_detection():
    global _DETECT_RUNNING
    _DETECT_RUNNING = False

# start detection for live road4 only when BACKGROUND_DET is enabled
if BACKGROUND_DET and 'road4' in STREAM_MANAGERS:
    start_detection_for_road('road4')
    atexit.register(stop_detection)


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


@app.route("/stream-status")
def stream_status():
    """Return current stream sources and timestamps for each road.

    Response example:
    {
      "Road 1": {"source": "...", "is_camera": true/false, "last_frame_time": 169...},
      ...
    }
    """
    status = {}
    try:
        for road, road_name in zip(APPROACH_ROADS, ROAD_NAMES):
            manager = STREAM_MANAGERS.get(road)
            if manager is None:
                status[road_name] = {"source": None, "is_camera": False, "last_frame_time": None}
                continue

            # Prefer manager.get_status() if present
            try:
                mstatus = manager.get_status()
            except Exception:
                mstatus = None

            if mstatus:
                status[road_name] = mstatus
            else:
                status[road_name] = {"source": getattr(manager, "source", None), "is_camera": False, "last_frame_time": None}

        return jsonify(status)
    except Exception as e:
        print("Error in /stream-status:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/raw-stream/<road>')
def raw_stream(road):
    """Return an MJPEG stream of annotated frames (runs detection per-frame).

    This endpoint performs one-shot detection per-frame and draws boxes. The
    camera / video source will be accessed only while a client is connected.
    """
    manager = STREAM_MANAGERS.get(road)
    if manager is None:
        return jsonify({'error': 'unknown road'}), 404

    def gen():
        boundary = b'--frame'
        try:
            while True:
                try:
                    frame = manager.next_frame()
                    if frame is None:
                        time.sleep(0.05)
                        continue


                    # Stream raw frames (no server-side drawing). Client overlays
                    # will poll /detection-boxes which is updated by tracker worker.
                    try:
                        annotated = frame
                        ret, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    except Exception:
                        time.sleep(0.02)
                        continue
                    if not ret:
                        time.sleep(0.02)
                        continue
                    jpeg = buf.tobytes()
                    yield boundary + b"\r\n"
                    yield b"Content-Type: image/jpeg\r\n"
                    yield f"Content-Length: {len(jpeg)}\r\n\r\n".encode('utf-8')
                    yield jpeg + b"\r\n"
                    # throttle MJPEG generation to STREAM_FPS for smooth client rendering
                    time.sleep(STREAM_INTERVAL)
                except GeneratorExit:
                    break
                except Exception:
                    time.sleep(0.05)
        finally:
            # nothing to explicitly clean up here; VideoStreamManager remains
            return

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/switch-source', methods=['POST', 'GET'])
def switch_source():
    """Switch the underlying source for a given road at runtime.

    Query / JSON params:
      road: e.g. 'road4'
      mode: 'camera' or 'file'

    This will release the previous VideoStreamManager (if any) and create a new
    one. Use with care: opening the camera will attempt to access the webcam.
    """
    try:
        # support both JSON POST and querystring GET
        data = request.get_json(silent=True) or request.args
        road = data.get('road')
        mode = data.get('mode')
        if not road or mode not in ('camera', 'file'):
            return jsonify({'error': 'invalid parameters, require road and mode=camera|file'}), 400

        if road not in APPROACH_ROADS:
            return jsonify({'error': 'unknown road'}), 400

        # Release old manager if present
        old = STREAM_MANAGERS.get(road)
        if old:
            try:
                old.release()
            except Exception:
                pass
            # give OS a short moment to release device handles (Windows)
            time.sleep(0.2)

        # Create new manager based on mode
        if mode == 'camera':
            # allow optional camera index param (query or JSON)
            idx = data.get('index')
            try:
                if idx is not None:
                    src = int(idx)
                else:
                    src = 0
            except Exception:
                return jsonify({'error': 'invalid index parameter'}), 400
        else:
            # Use default file path from video_common
            src = video_common.DEFAULT_VIDEO_PATHS.get(road)
            if not src or not os.path.exists(src):
                return jsonify({'error': f'video file for {road} not found'}), 404

        try:
            # for camera sources prefer a slightly higher background reader FPS for smoothness
            if mode == 'camera' and isinstance(src, int):
                STREAM_MANAGERS[road] = VideoStreamManager(src, max_fps=20.0)
            else:
                STREAM_MANAGERS[road] = VideoStreamManager(src)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        # if switching to camera mode, start background detection for this road to
        # provide low-latency cached boxes; if switching to file mode, stop camera worker
        try:
            if mode == 'camera':
                start_detection_for_road(road)
            else:
                stop_detection_for_road(road)
        except Exception:
            pass

        return jsonify({'ok': True, 'road': road, 'mode': mode, 'source': str(src)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/list-cameras')
def list_cameras():
    """Probe camera indices 0..6 and report which ones open and return frames.

    This endpoint is a diagnostic helper and should be fast. It does not keep
    camera handles open after probing.
    """
    results = []
    try:
        max_index = 6
        for i in range(0, max_index + 1):
            try:
                cap = cv2.VideoCapture(i)
                opened = bool(cap.isOpened())
                read_ok = False
                if opened:
                    ret, frame = cap.read()
                    read_ok = bool(ret and frame is not None)
                results.append({'index': i, 'opened': opened, 'read': read_ok})
            except Exception:
                results.append({'index': i, 'opened': False, 'read': False})
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/detection-boxes/<road>')
def detection_boxes(road):
    """Return a snapshot of detection boxes for a road.

    If background caching is enabled (BACKGROUND_DET=True) the cached boxes are
    returned. Otherwise a one-shot detection is performed on the latest frame.
    """
    try:
        # prefer cached entries if available (background worker or recently populated)
        with _DETECTION_LOCK:
            entry = _DETECTION_CACHE.get(road)
            if entry:
                return jsonify(entry)

        # one-shot detection path
        manager = STREAM_MANAGERS.get(road)
        if manager is None:
            return jsonify({"boxes": [], "width": None, "height": None, "ts": None}), 404

        frame = manager.next_frame()
        if frame is None:
            return jsonify({"boxes": [], "width": None, "height": None, "ts": None})

        try:
            with _INFERENCE_LOCK:
                dets = detect_vehicles(frame, model) if model is not None else []
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        boxes = []
        h, w = frame.shape[:2]
        for d in dets:
            if len(d) >= 5:
                # convert center-based xywh -> top-left for client overlay consistency
                cx, cy, bw, bh, cls = float(d[0]), float(d[1]), float(d[2]), float(d[3]), str(d[4])
                x = cx - (bw / 2.0)
                y = cy - (bh / 2.0)
                boxes.append({"x": float(x), "y": float(y), "w": float(bw), "h": float(bh), "class": cls})

        # cache this result briefly so raw_stream can use it when running
        with _DETECTION_LOCK:
            _DETECTION_CACHE[road] = {"boxes": boxes, "width": int(w), "height": int(h), "ts": time.time()}

        return jsonify({"boxes": boxes, "width": int(w), "height": int(h), "ts": time.time()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# detected-frame and detected-stream endpoints removed in revert


if __name__ == "__main__":
    print("🚦 AI Traffic Light Optimizer API Running on http://127.0.0.1:5000")
    print("✅ Routes available:")
    print("   → /vehicle-count  (for live counts)")
    print("   → /signal-status  (for signal decision)")
    # Run without the debugger to ensure no auto-reload/watchers will restart the process
    # when the controller writes logs. Enable threaded to handle concurrent requests.
    app.run(debug=False, port=5000, use_reloader=False, threaded=True)