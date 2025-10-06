from ultralytics import YOLO
from scipy.spatial import distance
from backend.detection.model_utils import get_centroid, vehicle_weights, DIST_THRESHOLD

# -----------------------------
# Load YOLOv8 model once
# -----------------------------
# Use local models folder
model = YOLO("models/best.pt")

# -----------------------------
# Detection Function
# -----------------------------
def detect_vehicles(frame, model):
    """Run YOLOv8 model on frame and return detections as (x, y, w, h, class_name)"""
    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        x, y, w, h = box.xywh[0].tolist()
        cls_id = int(box.cls[0].item())
        class_name = results.names[cls_id].lower()
        detections.append((x, y, w, h, class_name))
    return detections

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

        for obj_id, (old_c, old_class) in tracked_objects.items():
            if distance.euclidean((cx, cy), old_c) < DIST_THRESHOLD:
                new_tracked[obj_id] = ((cx, cy), class_name)
                found = True
                break

        if not found:
            next_vehicle_id += 1
            new_tracked[next_vehicle_id] = ((cx, cy), class_name)

    # Count and total weight
    count = len(new_tracked)
    total_weight = sum(vehicle_weights.get(cls, 2) for (_, cls) in [v for v in new_tracked.values()])

    return count, total_weight, new_tracked, next_vehicle_id
