import cv2

def get_video_frame(video_path):
    """
    Extracts a single frame from the given video file.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        frame (ndarray): The first frame of the video.

    Raises:
        RuntimeError: If the frame cannot be read.
    """
    # Initialize video capture object
    cap = cv2.VideoCapture(video_path)

    # Attempt to read the first frame
    ret, frame = cap.read()

    # Release the video capture resource
    cap.release()

    # Validate frame read success
    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame from video: {video_path}")

    return frame