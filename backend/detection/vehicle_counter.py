from scipy.spatial import distance
import cv2
from backend.detection.model_utils import get_centroid, vehicle_weights, DIST_THRESHOLD, draw_info

# -----------------------------
# Detection Function
# -----------------------------
def detect_vehicles(frame, model, show: bool = False, road_name: str = ""):
    """Run YOLOv8 model on frame and return detections as (x, y, w, h, class_name).

    This function is defensive: if frame is None or model returns no boxes,
    it returns an empty list instead of raising.
    """
    if frame is None:
        return []

    results = model(frame)[0]
    detections = []
    if not hasattr(results, "boxes") or len(results.boxes) == 0:
        return []

    detections_for_draw = []
    for box in results.boxes:
        # xywh may be a tensor of shape (1,4)
        xywh = box.xywh[0].tolist()
        x, y, w, h = xywh
        # safe access to class id
        try:
            cls_id = int(box.cls[0].item())
            class_name = results.names[cls_id].lower()
        except Exception:
            class_name = "unknown"

        detections.append((x, y, w, h, class_name))
        # prepare for optional drawing (center x,y and width/height, class)
        detections_for_draw.append((x, y, w, h, class_name))

    return detections

    # unreachable


# -----------------------------
# Tracking and Counting Logic
# -----------------------------
def update_tracks(detections, tracked_objects, next_vehicle_id):
    """
    detections: list of (x, y, w, h, class_name)
    tracked_objects: dict {obj_id: (centroid, class_name)}
    next_vehicle_id: int ID counter

    Returns:
        count: number of active vehicles on this road
        total_weight: sum of weights of active vehicles
        tracked_objects, next_vehicle_id: updated tracking data
    """
    new_tracked = {}

    # Match detections with previous tracked centroids
    for (x, y, w, h, class_name) in detections:
        cx, cy = get_centroid(x, y, w, h)
        found = False

        for obj_id, (old_c, old_class) in (tracked_objects or {}).items():
            if distance.euclidean((cx, cy), old_c) < DIST_THRESHOLD:
                new_tracked[obj_id] = ((cx, cy), class_name)
                found = True
                break

        if not found:
            next_vehicle_id += 1
            new_tracked[next_vehicle_id] = ((cx, cy), class_name)

    # Count and total weight
    count = len(new_tracked)
    total_weight = 0
    for (_, cls) in [v for v in new_tracked.values()]:
        try:
            total_weight += int(vehicle_weights.get(cls, 2))
        except Exception:
            total_weight += 2

    return count, total_weight, new_tracked, next_vehicle_id


def draw_detections_on_frame(frame, detections):
    """Return a copy of the frame with bounding boxes and class labels drawn.

    detections: iterable of (x, y, w, h, class_name) or (x,y,w,h,class_name,score)
    """
    if frame is None:
        return None

    out = frame.copy()
    for det in detections:
        if len(det) == 6:
            x, y, w, h, cls, score = det
            label = f"{cls}:{score:.2f}"
        else:
            x, y, w, h, cls = det
            label = f"{cls}"

        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        color = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return out


def create_trackers(frame, detections, tracker_type='MOSSE'):
    """Create OpenCV trackers for given detections.

    detections: iterable of (x,y,w,h, class_name) or (x,y,w,h,class_name,score)
    Returns: list of (tracker, class_name)
    """
    trackers = []
    if frame is None:
        return trackers

    for det in detections:
        try:
            if len(det) >= 5:
                x, y, w, h, cls = det[0], det[1], det[2], det[3], det[4]
            else:
                continue

            x1, y1, w_i, h_i = int(x), int(y), int(w), int(h)
            bbox = (x1, y1, w_i, h_i)

            # choose tracker
            if tracker_type == 'CSRT' and hasattr(cv2, 'TrackerCSRT_create'):
                tr = cv2.TrackerCSRT_create()
            elif hasattr(cv2, 'TrackerMOSSE_create'):
                tr = cv2.TrackerMOSSE_create()
            else:
                # fallback to MOSSE-like or raise
                tr = cv2.TrackerMOSSE_create()

            tr.init(frame, bbox)
            trackers.append((tr, cls))
        except Exception:
            continue

    return trackers


def update_trackers(frame, trackers):
    """Update trackers on the given frame.

    Returns: boxes, new_trackers
      boxes: list of (x,y,w,h,class_name)
      new_trackers: list of (tracker, class_name)
    """
    boxes = []
    new_trackers = []
    if frame is None:
        return boxes, new_trackers

    for tr, cls in trackers:
        try:
            ok, bbox = tr.update(frame)
            if not ok:
                continue
            x, y, w, h = bbox
            boxes.append((float(x), float(y), float(w), float(h), cls))
            new_trackers.append((tr, cls))
        except Exception:
            continue

    return boxes, new_trackers
