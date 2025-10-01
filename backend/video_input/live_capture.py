# ------------------------------------------------------------
# Module: live_capture.py
# Author: Shivam Paliwal
# Purpose: Capture real-time frames from webcam or video stream
# Usage: Used in run.py and app.py for live traffic detection
# ------------------------------------------------------------

import cv2

def get_live_frame(device_index=0):
    """
    Captures a single frame from the specified video device.

    Parameters:
    - device_index (int): Index of the video device (default = 0 for webcam)

    Returns:
    - frame (numpy.ndarray): Captured image frame

    Raises:
    - RuntimeError: If device cannot be opened or frame capture fails
    """
    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Video device {device_index} could not be opened.")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame from video device {device_index}.")

    return frame