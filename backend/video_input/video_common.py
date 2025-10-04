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

# ✅ Load multiple sampled frames from a video
def get_sampled_frames(video_path, num_frames=5, stride=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(num_frames):
        frame_idx = i * stride
        if frame_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

# ✅ Unified frame loader (video or live)
def get_frame(lane=None):
    if USE_VIDEO:
        path = VIDEO_PATHS.get(lane, VIDEO_PATHS["lane1"])
        return get_video_frame(path)
    else:
        from backend.video_input.live_capture import get_live_frame
        return get_live_frame()

# ✅ Display all frames from a video with 1-second delay
def get_all_video_frames(video_path):
    """
    Extracts and displays all frames from the given video file
    with a 1-second delay between each frame.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    while True:
        ret, frame = cap.read()

        # Stop if video ends or frame can't be read
        if not ret or frame is None:
            break

        # Display the current frame
        cv2.imshow('Video Frame', frame)

        # Wait for 1 second (1000 ms); exit if 'q' is pressed
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()