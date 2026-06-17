from ultralytics import YOLO
from typing import List, Tuple
import numpy as np

class DetectionEngine:
    """
    Wraps the YOLO model to provide consistent inference results.
    """

    def __init__(self, model_path: str):
        # Initialize YOLO model (supports NCNN/ONNX)
        self.model = YOLO(model_path)

    def predict(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Tuple[List[int], float, int]]:
        """
        Perform inference on a single frame.

        Returns:
            List of detections: [( [x1, y1, x2, y2], confidence, class_id ), ...]
        """
        # Run inference
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box coordinates
                b = box.xyxy[0].cpu().numpy().astype(int).tolist()
                # Confidence score
                conf = float(box.conf[0])
                # Class ID
                cls = int(box.cls[0])
                detections.append((b, conf, cls))

        return detections
