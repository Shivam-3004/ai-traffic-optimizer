# File: backend/video_input/video_common.py

from backend.video_input.stream_manager import VideoStreamManager
import os

USE_VIDEO = True

# Build absolute paths relative to project root to avoid CWD issues
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VIDEO_PATHS = {
    "road1": os.path.join(ROOT, "backend", "video_input", "videos", "t1.mp4"),
    "road2": os.path.join(ROOT, "backend", "video_input", "videos", "t2.mp4"),
    "road3": os.path.join(ROOT, "backend", "video_input", "videos", "t3.mp4"),
    "road4": os.path.join(ROOT, "backend", "video_input", "videos", "t4.mp4"),
}

# Create persistent stream managers where files exist; absent files are skipped but logged
STREAM_MANAGERS = {}
for road, path in VIDEO_PATHS.items():
    if not os.path.exists(path):
        print(f"Warning: video for road {road} not found at {path}")
        continue
    try:
        STREAM_MANAGERS[road] = VideoStreamManager(path)
    except Exception as e:
        print(f"Warning: couldn't open video for road {road} at {path}: {e}")


def get_live_video_frame(road):
    manager = STREAM_MANAGERS.get(road)
    if manager is None:
        return None
    return manager.next_frame()


def get_stream_managers():
    """Return the dictionary of stream managers (useful for threaded detectors)."""
    return STREAM_MANAGERS