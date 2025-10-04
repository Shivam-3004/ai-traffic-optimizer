# File: backend/video_input/video_common.py

from backend.video_input.stream_manager import VideoStreamManager

USE_VIDEO = True

VIDEO_PATHS = {
    "lane2": r"backend/video_input/videos/t1.mp4",
    "lane1": r"backend/video_input/videos/t2.mp4",
    "lane3": r"backend/video_input/videos/t3.mp4",
    "lane4": r"backend/video_input/videos/t4.mp4",
}

# ✅ Create persistent stream managers
STREAM_MANAGERS = {
    lane: VideoStreamManager(path)
    for lane, path in VIDEO_PATHS.items()
}

# ✅ Get one fresh frame per lane (simulated real-time)
def get_live_video_frame(lane):
    manager = STREAM_MANAGERS.get(lane)
    return manager.next_frame()