"""
src/core/person_detector.py — YOLOv8s person detection with ByteTrack.

Downloads yolov8s.pt (~22 MB) on first run.
"""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False


class PersonDetector:
    """YOLOv8s + ByteTrack person detector."""

    def __init__(self, model: str = "yolov8s.pt", conf: float = 0.4):
        if not _YOLO_OK:
            raise ImportError("pip install ultralytics")
        self._model = YOLO(model)
        self._conf  = conf

    def detect_and_track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv8s person detection + ByteTrack on frame.
        Returns list of {track_id: int, bbox: (x1,y1,x2,y2), confidence: float}.
        """
        results = self._model.track(
            frame,
            classes=[0],
            tracker="bytetrack.yaml",
            conf=self._conf,
            persist=True,
            verbose=False,
        )
        detections: List[Dict[str, Any]] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                tid  = int(box.id[0]) if box.id is not None else i
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "track_id":   tid,
                    "bbox":       (x1, y1, x2, y2),
                    "confidence": float(box.conf[0]),
                })
        return detections
