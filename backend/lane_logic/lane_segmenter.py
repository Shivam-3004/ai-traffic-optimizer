"""
Vehicle Counting & Lane Logic Module
- Splits frame into lane regions (A, B, C, D).
- Uses centroid tracking to avoid double-counting within a frame.
- Maintains lane-wise vehicle counters PER FRAME (resets after each frame).
- Plug in detection results here: detections = [(x,y,w,h), ...]
"""

import cv2
from scipy.spatial import distance

# Lane Definitions (ROIs)

lanes = {
    "lane1": (50, 100, 250, 400),   # (x1, y1, x2, y2)
    "lane2": (300, 100, 500, 400),
    "lane3": (550, 100, 750, 400),
    "lane4": (800, 100, 1000, 400)
}

# Tracking Setup

tracked_objects = {}  # vehicle_id : (centroid, lane)
next_vehicle_id = 0

# Helper Functions

def get_centroid(x, y, w, h):
    """Return centroid of bounding box"""
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return (cx, cy)


def which_lane(cx, cy):
    """Check which lane ROI contains centroid"""
    for lane, (x1, y1, x2, y2) in lanes.items():
        if x1 < cx < x2 and y1 < cy < y2:
            return lane
    return None


def update_tracks(detections):
    """
    detections = list of bounding boxes [(x,y,w,h), ...]
    Updates tracked objects per frame and resets count after every frame.
    """
    global tracked_objects, next_vehicle_id

    #  RESET COUNTS EVERY FRAME
    lane_counts = {lane: 0 for lane in lanes}

    new_tracked = {}
    for (x, y, w, h) in detections:
        cx, cy = get_centroid(x, y, w, h)
        lane = which_lane(cx, cy)
        if not lane:
            continue  # outside defined lanes

        found = False
        for obj_id, (old_c, old_lane) in tracked_objects.items():
            if distance.euclidean((cx, cy), old_c) < 30:  # threshold distance
                new_tracked[obj_id] = ((cx, cy), lane)
                found = True
                lane_counts[lane] += 1
                break

        if not found:
            # New vehicle → assign new ID
            next_vehicle_id += 1
            new_tracked[next_vehicle_id] = ((cx, cy), lane)
            lane_counts[lane] += 1

    tracked_objects = new_tracked
    return lane_counts

# MAIN LOOP (Integration with Detection)

if __name__ == "__main__":
    cap = cv2.VideoCapture("traffic.mp4")  # Change to 0 for webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # >>> This part must be filled by Detection Member <<<
        # detections = detection_model(frame)
        # Example dummy detections (x,y,w,h):
        detections = [(60, 120, 40, 40), (320, 150, 50, 50)]

        # Update tracking & lane counts PER FRAME
        counts = update_tracks(detections)
        print("Current Frame Lane Counts:", counts)

        # (Optional) Draw ROIs & counts on video frame
        for lane, (x1, y1, x2, y2) in lanes.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{lane}: {counts[lane]}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Traffic Counting", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()