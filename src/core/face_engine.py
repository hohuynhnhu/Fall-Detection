"""
src/core/face_engine.py — Nhận diện khuôn mặt dùng dlib

Yêu cầu (download từ dlib.net / GitHub):
  models/face/shape_predictor_68_face_landmarks.dat
  models/face/dlib_face_recognition_resnet_model_v1.dat

Install: pip install dlib
"""
from __future__ import annotations
import cv2
import numpy as np
import os
import pickle
import uuid
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SHAPE_PREDICTOR = os.path.join(_ROOT, "models", "face", "shape_predictor_68_face_landmarks.dat")
FACE_REC_MODEL  = os.path.join(_ROOT, "models", "face", "dlib_face_recognition_resnet_model_v1.dat")
FACE_DB_PATH    = os.path.join(_ROOT, "data", "profiles", "face_db.pkl")

RECOGNITION_THRESHOLD = 0.55   # Euclidean distance < threshold → match (0.6 = dlib default)


@dataclass
class FaceBox:
    x1: int; y1: int; x2: int; y2: int

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class RecognizedPerson:
    person_id:  str
    name:       str
    confidence: float   # 0–1, cao hơn = chắc hơn
    box:        FaceBox
    is_known:   bool = False


class FaceDatabase:
    """Lưu face encodings (128-d numpy) vào pickle."""

    def __init__(self, path: str = FACE_DB_PATH):
        self.path = path
        # {person_id: {"name": str, "role": str, "encodings": [np.ndarray], "added_at": float}}
        self.members: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.members = pickle.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.members, f)

    def add(self, person_id: str, name: str, role: str, encoding: np.ndarray):
        if person_id not in self.members:
            self.members[person_id] = {
                "name": name, "role": role,
                "encodings": [], "added_at": time.time(),
            }
        self.members[person_id]["encodings"].append(encoding)
        self.save()

    def remove(self, person_id: str) -> bool:
        if person_id in self.members:
            del self.members[person_id]
            self.save()
            return True
        return False

    def list_members(self) -> List[Dict]:
        return [
            {
                "person_id":    pid,
                "name":         d["name"],
                "role":         d["role"],
                "sample_count": len(d["encodings"]),
            }
            for pid, d in self.members.items()
        ]


class FaceEngine:
    """
    HOG face detector + dlib ResNet 128-d encoder.
    Nhận diện chạy mỗi `process_every_n` frames để giữ FPS.
    """

    def __init__(
        self,
        shape_predictor: str = SHAPE_PREDICTOR,
        face_rec_model:  str = FACE_REC_MODEL,
        db_path:         str = FACE_DB_PATH,
        threshold:       float = RECOGNITION_THRESHOLD,
        process_every_n: int   = 5,
    ):
        if not DLIB_AVAILABLE:
            raise ImportError("Cài dlib: pip install dlib")
        for p in (shape_predictor, face_rec_model):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Model không tồn tại: {p}\n"
                    "Tải từ: http://dlib.net/files/"
                )

        self.detector   = dlib.get_frontal_face_detector()
        self.predictor  = dlib.shape_predictor(shape_predictor)
        self.recognizer = dlib.face_recognition_model_v1(face_rec_model)
        self.db         = FaceDatabase(db_path)
        self.threshold  = threshold
        self._every_n   = process_every_n
        self._frame_cnt = 0
        self._cached:   List[RecognizedPerson] = []

    # ── Public ─────────────────────────────────────────────────────────────────

    def process(self, frame_bgr: np.ndarray) -> List[RecognizedPerson]:
        """Detect + nhận diện. Trả cache giữa các lần bỏ qua."""
        self._frame_cnt += 1
        if self._frame_cnt % self._every_n != 0:
            return self._cached

        # Detect ở ảnh nhỏ (0.5×) cho nhanh
        small   = cv2.resize(frame_bgr, (0, 0), fx=0.5, fy=0.5)
        rgb_s   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rects_s = self.detector(rgb_s, 0)

        rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results  = []
        for r in rects_s:
            r2 = dlib.rectangle(r.left() * 2, r.top() * 2,
                                r.right() * 2, r.bottom() * 2)
            box      = FaceBox(r2.left(), r2.top(), r2.right(), r2.bottom())
            shape    = self.predictor(rgb_full, r2)
            encoding = np.array(self.recognizer.compute_face_descriptor(rgb_full, shape))
            results.append(self._match(encoding, box))

        self._cached = results
        return results

    def enroll(
        self,
        frame_bgr: np.ndarray,
        name:      str,
        role:      str = "family",
        person_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Enroll khuôn mặt lớn nhất trong frame vào database.
        Trả về person_id nếu thành công, None nếu không thấy mặt.
        """
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rects = self.detector(rgb, 0)
        if not rects:
            return None
        rect     = max(rects, key=lambda r: r.width() * r.height())
        shape    = self.predictor(rgb, rect)
        encoding = np.array(self.recognizer.compute_face_descriptor(rgb, shape))
        pid      = person_id or str(uuid.uuid4())[:8]
        self.db.add(pid, name, role, encoding)
        return pid

    def remove_member(self, person_id: str) -> bool:
        return self.db.remove(person_id)

    def list_members(self) -> List[Dict]:
        return self.db.list_members()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _match(self, encoding: np.ndarray, box: FaceBox) -> RecognizedPerson:
        best_id   = "unknown"
        best_name = "Không nhận ra"
        best_dist = float("inf")

        for pid, data in self.db.members.items():
            for enc in data["encodings"]:
                dist = float(np.linalg.norm(encoding - enc))
                if dist < best_dist:
                    best_dist = dist
                    best_id   = pid
                    best_name = data["name"]

        if best_dist > self.threshold:
            return RecognizedPerson("unknown", "Không nhận ra", 0.0, box, False)

        confidence = max(0.0, 1.0 - best_dist / self.threshold)
        return RecognizedPerson(best_id, best_name, confidence, box, True)
