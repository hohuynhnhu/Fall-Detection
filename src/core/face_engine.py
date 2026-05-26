"""
src/core/face_engine.py — Person detection + face recognition.

YOLOv8s + ByteTrack (PersonDetector) → InsightFace buffalo_l (ArcFace, 512-d).
Cosine similarity threshold: 0.35.

Install: pip install ultralytics insightface onnxruntime
"""
from __future__ import annotations
import cv2
import numpy as np
import os
import pickle
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from .appearance_tracker import AppearanceSnapshot
    _APPEARANCE_OK = True
except ImportError:
    _APPEARANCE_OK = False
    AppearanceSnapshot = None  # type: ignore

try:
    from .person_detector import PersonDetector
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

try:
    from .face_recognizer_if import FaceRecognizer, cosine_similarity
    _INSIGHTFACE_OK = True
except ImportError:
    _INSIGHTFACE_OK = False

_ROOT         = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FACE_DB_PATH  = os.path.join(_ROOT, "data", "profiles", "face_db.pkl")

RECOGNITION_THRESHOLD = 0.35   # cosine similarity >= threshold → match
_EMBEDDING_DIM        = 512    # InsightFace ArcFace


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
    is_patient: bool = False
    track_id:   int  = -1


class FaceDatabase:
    """Stores 512-d InsightFace embeddings in a local pickle file."""

    def __init__(self, path: str = FACE_DB_PATH):
        self.path = path
        # {person_id: {name, role, is_patient, encodings: [ndarray(512)], added_at}}
        self.members: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            # Filter out incompatible dlib 128-d embeddings automatically
            valid: Dict[str, Any] = {}
            for pid, d in data.items():
                good = [
                    e for e in d.get("encodings", [])
                    if isinstance(e, np.ndarray) and e.shape == (_EMBEDDING_DIM,)
                ]
                if good:
                    d["encodings"] = good
                    valid[pid] = d
            self.members = valid
        except Exception:
            self.members = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.members, f)

    def add(self, person_id: str, name: str, role: str,
            encoding: np.ndarray, is_patient: bool = False):
        if person_id not in self.members:
            self.members[person_id] = {
                "name": name, "role": role,
                "is_patient": is_patient,
                "encodings": [], "added_at": time.time(),
            }
        else:
            self.members[person_id]["is_patient"] = is_patient
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
                "role":         d.get("role", "family"),
                "is_patient":   d.get("is_patient", False),
                "sample_count": len(d["encodings"]),
            }
            for pid, d in self.members.items()
        ]


class FaceEngine:
    """
    Person detection via YOLOv8s + ByteTrack.
    Face recognition via InsightFace buffalo_l (ArcFace, 512-d, cosine similarity).
    """

    def __init__(
        self,
        yolo_model:      str   = "yolov8s.pt",
        yolo_conf:       float = 0.4,
        det_size:        tuple = (320, 320),
        db_path:         str   = FACE_DB_PATH,
        threshold:       float = RECOGNITION_THRESHOLD,
        process_every_n: int   = 5,
    ):
        if not _YOLO_OK:
            raise ImportError("pip install ultralytics")
        if not _INSIGHTFACE_OK:
            raise ImportError("pip install insightface onnxruntime")

        self._person_detector = PersonDetector(model=yolo_model, conf=yolo_conf)
        self._face_rec        = FaceRecognizer(det_size=det_size)
        self.db               = FaceDatabase(db_path)
        self.threshold        = threshold
        self._every_n         = process_every_n
        self._frame_cnt       = 0
        self._cached:         List[RecognizedPerson] = []
        self._family_manager  = None

        # Per-track attempt counters for process_frame()
        self._track_attempts: Dict[int, int] = {}
        self._MAX_ATTEMPTS = 10

    def set_family_manager(self, fm) -> None:
        """Inject FamilyManager — _match() will use network gallery instead of local DB."""
        self._family_manager = fm

    # ── Public ─────────────────────────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> List[RecognizedPerson]:
        """
        Full pipeline: YOLOv8s detect+track → InsightFace per-person crop.
        Runs every _every_n frames; caches results in between.
        Each track_id is attempted at most _MAX_ATTEMPTS times.
        """
        self._frame_cnt += 1
        if self._frame_cnt % self._every_n != 0:
            return self._cached

        detections = self._person_detector.detect_and_track(frame_bgr)
        h, w       = frame_bgr.shape[:2]
        results:   List[RecognizedPerson] = []
        active_ids = {d["track_id"] for d in detections}

        for det in detections:
            tid  = det["track_id"]
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            crop     = frame_bgr[y1:y2, x1:x2]
            face_box = FaceBox(x1, y1, x2, y2)

            if crop.size == 0 or self._track_attempts.get(tid, 0) >= self._MAX_ATTEMPTS:
                cached = next((r for r in self._cached if r.track_id == tid), None)
                if cached:
                    results.append(cached)
                continue

            faces = self._face_rec.get_faces_in_crop(crop)
            self._track_attempts[tid] = self._track_attempts.get(tid, 0) + 1

            if not faces:
                results.append(
                    RecognizedPerson("unknown", "Không nhận ra", 0.0, face_box,
                                     False, False, tid))
                continue

            best = max(
                faces,
                key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
            )
            results.append(self._match(best["embedding"], face_box, tid))

        # Cleanup stale per-track state
        for tid in list(self._track_attempts):
            if tid not in active_ids:
                self._track_attempts.pop(tid, None)

        self._cached = results
        return results

    def run_face_recognition(
        self,
        frame_bgr: np.ndarray,
        box:       tuple,
    ) -> Optional[RecognizedPerson]:
        """
        Crop person bbox, run InsightFace on crop, return matched identity.
        Returns None when crop is too small or no face detected.
        """
        x1, y1, x2, y2 = (int(v) for v in box)
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            return None

        fh, fw = frame_bgr.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(fw, x2); y2 = min(fh, y2)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        faces = self._face_rec.get_faces_in_crop(crop)
        if not faces:
            return None

        best = max(
            faces,
            key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
        )
        return self._match(best["embedding"], FaceBox(x1, y1, x2, y2))

    def enroll(
        self,
        frame_bgr:  np.ndarray,
        name:       str,
        role:       str  = "family",
        person_id:  Optional[str] = None,
        is_patient: bool = False,
    ) -> Optional[str]:
        """
        Detect face in frame, extract 512-d InsightFace embedding, save to local DB.
        Returns person_id on success, None if no face found.
        """
        embedding = self._face_rec.get_embedding_from_image(frame_bgr)
        if embedding is None:
            return None
        pid = person_id or str(uuid.uuid4())[:8]
        self.db.add(pid, name, role, embedding, is_patient=is_patient)
        return pid

    def enroll_from_bytes(
        self,
        image_bytes: bytes,
        name:        str,
        role:        str  = "family",
        person_id:   Optional[str] = None,
        is_patient:  bool = False,
    ) -> Optional[str]:
        """Enroll face from raw image bytes (JPEG/PNG)."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None
        return self.enroll(frame_bgr, name=name, role=role,
                           person_id=person_id, is_patient=is_patient)

    def remove_member(self, person_id: str) -> bool:
        return self.db.remove(person_id)

    def list_members(self) -> List[Dict]:
        return self.db.list_members()

    def is_patient(self, person_id: str) -> bool:
        member = self.db.members.get(person_id)
        return bool(member and member.get("is_patient", False))

    def extract_appearance(
        self,
        frame_bgr:      np.ndarray,
        box:            tuple,
        pose_landmarks: Optional[object] = None,
    ) -> "AppearanceSnapshot":
        """
        Build AppearanceSnapshot from a person bounding box.
        Splits crop into vertical thirds for upper/lower HSV histograms.
        Shape metrics come from MediaPipe pose_landmarks when provided.
        """
        if not _APPEARANCE_OK:
            raise ImportError("appearance_tracker module not available")

        x1, y1, x2, y2 = (int(v) for v in box)
        fh, fw = frame_bgr.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(fw, x2); y2 = min(fh, y2)
        crop   = frame_bgr[y1:y2, x1:x2]
        bh     = crop.shape[0]
        aspect = (y2 - y1) / max(x2 - x1, 1)

        _EMPTY = np.zeros(18 * 8, dtype=np.float32)

        def _hist(region: np.ndarray) -> np.ndarray:
            if region.size == 0 or region.shape[0] < 2 or region.shape[1] < 2:
                return _EMPTY.copy()
            hsv  = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist.flatten().astype(np.float32)

        third      = max(1, bh // 3)
        hist_upper = _hist(crop[:third])
        hist_lower = _hist(crop[2 * third:])

        shoulder_width = 0.0
        body_height    = 0.0
        if pose_landmarks is not None:
            try:
                lm   = pose_landmarks.landmark
                ls   = lm[11]; rs   = lm[12]
                nose = lm[0];  la   = lm[27]; ra = lm[28]
                shoulder_width = abs(ls.x - rs.x) * fw
                ankle_y        = ((la.y + ra.y) / 2) * fh
                body_height    = abs(ankle_y - nose.y * fh)
            except Exception:
                pass

        return AppearanceSnapshot(
            color_hist_upper = hist_upper,
            color_hist_lower = hist_lower,
            shoulder_width   = shoulder_width,
            body_height      = body_height,
            aspect_ratio     = aspect,
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _match(
        self,
        embedding: np.ndarray,
        box:       FaceBox,
        track_id:  int = -1,
    ) -> RecognizedPerson:
        # Priority: FamilyManager (network-backed gallery from backend)
        if self._family_manager is not None:
            result = self._family_manager.match_person(embedding)
            if result:
                return RecognizedPerson(
                    person_id  = result["person_id"],
                    name       = result["name"],
                    confidence = result["confidence"],
                    box        = box,
                    is_known   = True,
                    is_patient = result["is_patient"],
                    track_id   = track_id,
                )
            return RecognizedPerson(
                "unknown", "Không nhận ra", 0.0, box, False, False, track_id)

        # Fallback: local pickle DB (enrolled via desktop UI)
        best_id    = "unknown"
        best_name  = "Không nhận ra"
        best_score = -1.0
        for pid, data in self.db.members.items():
            for enc in data["encodings"]:
                score = cosine_similarity(embedding, enc)
                if score > best_score:
                    best_score = score
                    best_id    = pid
                    best_name  = data["name"]
        if best_score < self.threshold:
            return RecognizedPerson(
                "unknown", "Không nhận ra", 0.0, box, False, False, track_id)
        is_pat = bool(self.db.members.get(best_id, {}).get("is_patient", False))
        return RecognizedPerson(
            best_id, best_name, best_score, box, True, is_pat, track_id)
