# File: backend/video_input/stream_manager.py

import cv2

class VideoStreamManager:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to start
            ret, frame = self.cap.read()
        return frame

    def release(self):
        self.cap.release()