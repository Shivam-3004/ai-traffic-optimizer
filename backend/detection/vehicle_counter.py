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
