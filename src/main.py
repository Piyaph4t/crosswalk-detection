import cv2
from cv2 import imshow, waitKey
import numpy as np


def letterbox_resize(image, target_size=640):
    """Resize proportionally and pad to square"""
    h, w = image.shape[:2]

    # Calculate scaling factor to fit longest side to target
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize proportionally
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create square canvas and center the image (letterbox)
    canvas = np.full(
        (target_size, target_size, 3), 114, dtype=np.uint8
    )  # 114 = gray padding

    # Place resized image in center
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


capture = cv2.VideoCapture("test.mp4")

while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        frame = letterbox_resize(frame, 640)
        imshow("Capture", frame)

    if waitKey(1) == ord("q") or ret == False:
        break

capture.release()
cv2.destroyAllWindows()
