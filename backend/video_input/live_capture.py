# ------------------------------------------------------------
# Module: live_capture.py
# Author: Shivam Paliwal
# Purpose: Capture real-time frames from webcam or video stream
# Usage: Used in run.py and app.py for live traffic detection
# ------------------------------------------------------------

import cv2
import threading
import time
import logging
from typing import Optional, Tuple, Callable

logger = logging.getLogger("live_capture")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(ch)


class LiveCapture:
    """Continuously capture frames from a camera in a background thread.

    Features:
    - Background thread continuously reads frames to reduce read latency.
    - Optional callback on_frame to run model inference and return detections.
    - Annotates frames with bounding boxes and labels if detections are provided.
    - Exposes get_frame() to retrieve the latest annotated frame.
    """

    def _init_(self, device_index: int = 0, on_frame: Optional[Callable] = None):
        """Create and start the capture thread.

        Args:
            device_index: OpenCV device index (0 for default webcam) or video URL/path.
            on_frame: Optional callable(frame) -> List[(x,y,w,h,class_name,score)] used to
                      perform inference. If provided, its results are drawn on the frame.
        """
        self.device = device_index
        self.on_frame = on_frame
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Video device {device_index} could not be opened.")

        self._lock = threading.Lock()
        self._running = True
        self._frame = None  # latest raw frame
        self._annotated = None  # latest annotated frame (BGR)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.debug(f"LiveCapture started on device={device_index}")

    def _run(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # small sleep to avoid tight loop when camera disconnects
                logger.debug("Frame read failed, retrying in 0.1s")
                time.sleep(0.1)
                continue

            annotated = frame.copy()
            # If a callback is provided, run model inference and draw boxes
            if self.on_frame is not None:
                try:
                    detections = self.on_frame(frame) or []
                    # Expected detection format: iterable of (x, y, w, h, class_name, score)
                    for det in detections:
                        if len(det) == 6:
                            x, y, w, h, cls, score = det
                        elif len(det) == 5:
                            x, y, w, h, cls = det
                            score = None
                        else:
                            # unexpected shape; skip
                            continue

                        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                        color = (0, 255, 0)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls}" + (f":{score:.2f}" if score is not None else "")
                        # put label background
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                        cv2.putText(annotated, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                except Exception as e:
                    logger.exception(f"Error running on_frame callback: {e}")

            with self._lock:
                self._frame = frame
                self._annotated = annotated

        # release capture when stopping
        try:
            self.cap.release()
        except Exception:
            pass
        logger.debug("LiveCapture thread exiting")

    def get_frame(self, annotated: bool = True) -> Optional[object]:
        """Return the latest frame.

        Args:
            annotated: If True and annotations are available, return annotated frame.
        Returns:
            BGR ndarray or None if no frame available yet.
        """
        with self._lock:
            return self._annotated.copy() if (annotated and self._annotated is not None) else (self._frame.copy() if self._frame is not None else None)

    def stop(self):
        """Stop the background thread and release the capture device."""
        self._running = False
        # wait for thread to finish
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def get_live_frame(device_index=0, model=None, class_names: Optional[dict] = None):
    """Backward-compatible helper: capture one frame and optionally run model inference.

    If model is provided, it should be a callable that accepts a BGR numpy array and
    returns a list of detections in the format (x, y, w, h, class_name, score) or
    (x, y, w, h, class_name).
    """
    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Video device {device_index} could not be opened.")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame from video device {device_index}.")

    annotated = frame.copy()
    if model is not None:
        try:
            detections = model(frame) or []
            for det in detections:
                if len(det) == 6:
                    x, y, w, h, cls, score = det
                elif len(det) == 5:
                    x, y, w, h, cls = det
                    score = None
                else:
                    continue

                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{cls}" + (f":{score:.2f}" if score is not None else "")
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        except Exception as e:
            logger.exception(f"Error running model on single frame: {e}")

    return annotated


if __name__ == "_main_":
    # Simple demo: open webcam, run capture, show annotated frames. This demo does not
    # perform model inference by default; pass a model callable to LiveCapture to enable it.
    try:
        lc = LiveCapture(1)
        print("Press 'q' in the window to quit. Showing live camera (no model inference).")
        while True:
            frame = lc.get_frame(annotated=False)
            if frame is None:
                time.sleep(0.05)
                continue
            cv2.imshow("Live Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        try:
            lc.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()