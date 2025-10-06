import cv2

# -----------------------------
# Road Video Sources
# (Each road has its own video feed)
# -----------------------------
roads = {
    "roadA": "backend/video_input/videos/t1.mp4",
    "roadB": "backend/video_input/videos/t2.mp4",
    "roadC": "backend/video_input/videos/t3.mp4",
    "roadD": "backend/video_input/videos/t4.mp4"
}

# -----------------------------
# Vehicle Weights
# -----------------------------
vehicle_weights = {
    "ambulance": 4,
    "bus": 5,
    "car": 3,
    "motorcycle": 1,
    "truck": 5,
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


def draw_info(frame, count, weight_sum, detections, road_name):
    """
    Draw detection boxes and display count & total weight for this road.
    """
    h, w, _ = frame.shape
    overlay_text = f"{road_name}: {count} | Weight={weight_sum}"

    # Draw label bar at top of frame
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, overlay_text, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw detection boxes
    for (x, y, bw, bh, cls) in detections:
        cv2.rectangle(frame,
                      (int(x - bw / 2), int(y - bh / 2)),
                      (int(x + bw / 2), int(y + bh / 2)),
                      (255, 0, 0), 2)
        cv2.putText(frame, cls,
                    (int(x - bw / 2), int(y - bh / 2) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame
