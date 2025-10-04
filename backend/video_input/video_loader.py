import cv2

VIDEO_PATHS = {
    "lane1": "backend/video_input/videos/t1.mp4",
    "lane2": "backend/video_input/videos/t2.mp4",
    "lane3": "backend/video_input/videos/t3.mp4",
    "lane4": "backend/video_input/videos/t4.mp4",
}

video_caps = {
    lane: cv2.VideoCapture(path)
    for lane, path in VIDEO_PATHS.items()
}

def get_next_frame(lane):
    cap = video_caps[lane]
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return frame