import cv2
from typing import List, Tuple
import numpy as np

def draw_detections(frame: np.ndarray, detections: List[Tuple[List[int], float, int]]) -> np.ndarray:
    """
    Draws bounding boxes and labels on the frame.
    """
    output_frame = frame.copy()

    for bbox, conf, cls in detections:
        x1, y1, x2, y2 = bbox

        # Define color based on class (simple mapping)
        color = (0, 255, 0) if cls == 0 else (255, 0, 0) # Green for person, blue for others
        label = f"ID:{cls} {conf:.2f}"

        # Draw Bounding Box
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

        # Draw Label
        cv2.putText(
            output_frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return output_frame
