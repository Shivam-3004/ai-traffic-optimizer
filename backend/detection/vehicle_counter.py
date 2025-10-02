from ultralytics import YOLO
from scipy.spatial import distance
from backend.detection.model_utils import get_centroid, which_lane, lanes, vehicle_weights, DIST_THRESHOLD

# Load YOLOv8 model once
model = YOLO("backend/detection/models/best.pt")

def detect_vehicles(frame, model):
    """Run YOLOv8 model on frame and return detections as (x,y,w,h,class_name)"""
    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        x, y, w, h = box.xywh[0].tolist()
        cls_id = int(box.cls[0].item())
        class_name = results.names[cls_id]
        detections.append((x, y, w, h, class_name))
    return detections

def update_tracks(detections, tracked_objects, next_vehicle_id):
    """
    detections = list of (x,y,w,h,class_name)
    tracked_objects = dict of obj_id : (centroid, lane, class_name)
    next_vehicle_id = counter for IDs

    Returns:
      lane_counts = {lane: active vehicles}
      lane_weights = {lane: weight sum of active vehicles}
      tracked_objects, next_vehicle_id (updated)
    """
    lane_counts = {lane: 0 for lane in lanes}
    lane_weights = {lane: 0 for lane in lanes}
    new_tracked = {}

    for (x, y, w, h, class_name) in detections:
        cx, cy = get_centroid(x, y, w, h)
        lane = which_lane(cx, cy)
        if not lane:
            continue

        found = False
        for obj_id, (old_c, old_lane, old_class) in tracked_objects.items():
            if distance.euclidean((cx, cy), old_c) < DIST_THRESHOLD:
                new_tracked[obj_id] = ((cx, cy), lane, class_name)
                found = True
                break

        if not found:
            next_vehicle_id += 1
            new_tracked[next_vehicle_id] = ((cx, cy), lane, class_name)

    for obj_id, (c, lane, cls) in new_tracked.items():
        lane_counts[lane] += 1
        lane_weights[lane] += vehicle_weights.get(cls, 2)

    return lane_counts, lane_weights, new_tracked, next_vehicle_id
