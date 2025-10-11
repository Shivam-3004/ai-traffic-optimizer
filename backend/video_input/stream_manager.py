# File: backend/video_input/stream_manager.py
import threading
import cv2
import time

class VideoStreamManager:
    """Manage a video source which can be a file path or an integer camera index.

    For file-based sources we loop back to the start when EOF is reached. For
    camera sources (int) we do not attempt to rewind; instead we return None
    briefly when reads fail which allows callers to handle transient failures.
    """
    def __init__(self, source):
        # Accept either an int (camera index) or a filesystem path/URL
        self.source = source
        # Try multiple backends for camera indices on Windows which often
        # need DirectShow or Media Foundation to access external webcams.
        self.cap = None
        tried = []
        def try_open(src, api_preference=None):
            try:
                if api_preference is None:
                    c = cv2.VideoCapture(src)
                else:
                    c = cv2.VideoCapture(src, api_preference)
                return c
            except Exception:
                return None

        # If source looks like a camera index (int), try DSHOW -> MSMF -> default
        if isinstance(source, int):
            # prefer DirectShow on Windows
            for api in [getattr(cv2, 'CAP_DSHOW', None), getattr(cv2, 'CAP_MSMF', None), None]:
                if api is None:
                    c = try_open(source, None)
                else:
                    c = try_open(source, api)
                tried.append((source, api, bool(c and c.isOpened())))
                if c is not None and c.isOpened():
                    self.cap = c
                    break
        else:
            # file path or URL: default open
            self.cap = try_open(source, None)

        if self.cap is None or not self.cap.isOpened():
            # helpful error message includes the source repr and backends tried
            raise RuntimeError(f"Cannot open video source: {repr(source)}; tried: {tried}")
        self.lock = threading.Lock()
        # determine if this is a camera (int) or file-like source
        self._is_camera = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
        # timestamp of last successful frame read (seconds since epoch) or None
        self.last_frame_time = None

    def next_frame(self):
        with self.lock:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                try:
                    self.last_frame_time = time.time()
                except Exception:
                    pass
                return frame

            # If source is a file, attempt to loop back to start once
            if not self._is_camera:
                try:
                    # reset to start and read again
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        return frame
                except Exception:
                    pass

            # For cameras or after a failed file read, return None and let caller decide
            return None

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def get_status(self):
        """Return a small status dict: source, last_frame_time (or None), and whether this is a camera."""
        return {
            "source": self.source,
            "is_camera": bool(self._is_camera),
            "last_frame_time": int(self.last_frame_time) if self.last_frame_time is not None else None,
        }