# ------------------------------------------------------------
# File: backend/video_input/video_common.py
# Purpose: Centralized video paths and frame loader
# ------------------------------------------------------------

import cv2

# ✅ Toggle between video and live webcam
USE_VIDEO = True

# ✅ Define per-lane video paths
VIDEO_PATHS = {
    "lane2": r"backend/video_input/videos/t1.mp4",
    "lane1": r"backend/video_input/videos/t2.mp4",
    "lane3": r"backend/video_input/videos/t3.mp4",
    "lane4": r"backend/video_input/videos/t4.mp4",
}

# ✅ Load one frame from a video
def get_video_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame from {video_path}")
    return frame

# ✅ Unified frame loader (video or live)
def get_frame(lane=None):
    if USE_VIDEO:
        path = VIDEO_PATHS.get(lane, VIDEO_PATHS["lane1"])
        return get_video_frame(path)
    else:
        from backend.video_input.live_capture import get_live_frame
        return get_live_frame()