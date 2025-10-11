# File: backend/video_input/video_common.py

from backend.video_input.stream_manager import VideoStreamManager
import os

USE_VIDEO = True

# Build absolute paths relative to project root to avoid CWD issues
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Default video file paths (file-based sources)
DEFAULT_VIDEO_PATHS = {
    "road1": os.path.join(ROOT, "backend", "video_input", "videos", "t1.mp4"),
    "road2": os.path.join(ROOT, "backend", "video_input", "videos", "t2.mp4"),
    "road3": os.path.join(ROOT, "backend", "video_input", "videos", "t3.mp4"),
    "road4": os.path.join(ROOT, "backend", "video_input", "videos", "t4.mp4"),
}

# VIDEO_SOURCES is the canonical mapping of road -> source. A source may be:
# - a filesystem path to a video file
# - an integer camera index (0, 1, ...)
# We try to load `video_sources.json` which is an easy-to-edit mapping. If the
# file is absent, fall back to DEFAULT_VIDEO_PATHS and a camera fallback for road4.
VIDEO_SOURCES = {}
config_path = os.path.join(os.path.dirname(__file__), "video_sources.json")
if os.path.exists(config_path):
    try:
        import json

        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for road, src in data.items():
            # support a small "camera:N" syntax in the JSON for ease of editing
            if isinstance(src, str) and src.startswith("camera:"):
                try:
                    VIDEO_SOURCES[road] = int(src.split("camera:", 1)[1])
                except Exception:
                    VIDEO_SOURCES[road] = src
            else:
                VIDEO_SOURCES[road] = src
    except Exception as e:
        print(f"Warning: failed to parse {config_path}: {e}")
        # fallback to default paths below

if not VIDEO_SOURCES:
    for road, path in DEFAULT_VIDEO_PATHS.items():
        VIDEO_SOURCES[road] = path

    # If a video file for road4 is missing, fall back to the default camera (index 0).
    if not os.path.exists(VIDEO_SOURCES["road4"]):
        print(f"Info: Video for road4 not found at {VIDEO_SOURCES['road4']}; falling back to camera index 0")
        VIDEO_SOURCES["road4"] = 0

# Create persistent stream managers where sources are available
STREAM_MANAGERS = {}
for road, source in VIDEO_SOURCES.items():
    try:
        STREAM_MANAGERS[road] = VideoStreamManager(source)
    except Exception as e:
        print(f"Warning: couldn't open source for road {road} ({repr(source)}): {e}")


def get_live_video_frame(road):
    manager = STREAM_MANAGERS.get(road)
    if manager is None:
        return None
    return manager.next_frame()


def get_stream_managers():
    """Return the dictionary of stream managers (useful for threaded detectors)."""
    return STREAM_MANAGERS