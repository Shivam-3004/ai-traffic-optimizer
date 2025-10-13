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
    def __init__(self, source, max_fps: float = 15.0):
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
        # instance attributes
        self.lock = threading.Lock()
        # determine if this is a camera (int) or file-like source
        self._is_camera = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
        # timestamp of last successful frame read (seconds since epoch) or None
        self.last_frame_time = None
        # background reader state
        self._running = True
        # maximum frames per second to read in background (reduce CPU)
        try:
            self.max_fps = float(max_fps) if max_fps and float(max_fps) > 0 else 15.0
        except Exception:
            self.max_fps = 15.0
        self._frame_interval = 1.0 / float(self.max_fps) if self.max_fps > 0 else 0.066
        self._last_frame = None
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

    def next_frame(self):
        # Return the most recent frame read by the background reader.
        with self.lock:
            if self._last_frame is None:
                return None
            # return a copy to avoid callers mutating the cached frame
            try:
                return self._last_frame.copy()
            except Exception:
                return self._last_frame

    def release(self):
        # stop reader thread then release capture
        try:
            self._running = False
            if hasattr(self, '_reader_thread') and self._reader_thread.is_alive():
                self._reader_thread.join(timeout=1.0)
        except Exception:
            pass
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

    def _reader(self):
        """Background thread that continuously reads frames and updates _last_frame.

        This reduces latency for callers because reads are non-blocking and
        the latest frame is always available immediately.
        """
        while getattr(self, '_running', False):
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # for file-like sources, try rewinding once
                    if not self._is_camera:
                        try:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = self.cap.read()
                        except Exception:
                            ret = False
                    if not ret:
                        # wait a short time before retrying to avoid tight loop
                        time.sleep(min(0.1, self._frame_interval))
                        continue

                with self.lock:
                    self._last_frame = frame
                    try:
                        self.last_frame_time = time.time()
                    except Exception:
                        pass
                # throttle reader to target FPS
                try:
                    time.sleep(self._frame_interval)
                except Exception:
                    pass
            except Exception:
                time.sleep(0.02)
                continue