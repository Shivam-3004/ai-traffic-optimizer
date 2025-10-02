import cv2

# -----------------------------
# Lane Definitions (ROIs)
# -----------------------------
lanes = {
    "lane1": (50, 100, 250, 400),
    "lane2": (300, 100, 500, 400),
    "lane3": (550, 100, 750, 400),
    "lane4": (800, 100, 1000, 400)
}

# -----------------------------
# Vehicle Weights
# -----------------------------
vehicle_weights = {
    "Auto_rickshaw": 3,
    "Bike": 1,
    "Bus": 5,
    "Car": 3,
    "HCV": 5,
    "LCV": 4,
    "Toto": 2,
}

# -----------------------------
# Tracking Threshold
# -----------------------------
DIST_THRESHOLD = 30  # Centroid distance for tracking

# -----------------------------
# Helper Functions
# -----------------------------
def get_centroid(x, y, w, h):
    """Return centroid of bounding box"""
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return cx, cy

def which_lane(cx, cy):
    """Check which lane ROI contains centroid"""
    for lane, (x1, y1, x2, y2) in lanes.items():
        if x1 < cx < x2 and y1 < cy < y2:
            return lane
    return None

def draw_info(frame, counts, weights, detections):
    """Draw ROIs, counts, weights, and detections"""
    for lane, (x1, y1, x2, y2) in lanes.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{lane}: {counts[lane]} | W={weights[lane]}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    for (x, y, w, h, cls) in detections:
        cv2.rectangle(frame, (int(x - w/2), int(y - h/2)),
                      (int(x + w/2), int(y + h/2)), (255, 0, 0), 2)
        cv2.putText(frame, cls, (int(x - w/2), int(y - h/2) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame
