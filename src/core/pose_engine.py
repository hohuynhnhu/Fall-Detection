"""
src/core/pose_engine.py

MediaPipe Pose + YOLOv8 Pose fusion
- MediaPipe: 33 keypoints, primary model
- YOLO Pose: 17 keypoints, optional
- Fusion: confidence-weighted average khi cả 2 detect được
"""
from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import defaultdict
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class PersonData:
    person_id:   int           # ID tracking từ YOLO ByteTrack
    metrics:     BodyMetrics
    raw_kp:      np.ndarray    # (200,)
    bbox:        tuple         # (x1, y1, x2, y2)

@dataclass
class BodyMetrics:
    shoulder_y: float   = 0.0
    hip_y: float        = 0.0
    ankle_y: float      = 0.0
    knee_l_y: float     = 0.0
    knee_r_y: float     = 0.0
    nose_y: float       = 0.0
    wrist_l_y: float    = 0.0
    wrist_r_y: float    = 0.0
    body_angle: float   = 0.0    # 0° = đứng thẳng, 90° = nằm ngang
    aspect_ratio: float = 1.5    # H/W bbox. < 0.6 → nằm
    center_x: float     = 0.0
    center_y: float     = 0.0
    bbox_w: float       = 0.0
    bbox_h: float       = 0.0
    confidence: float   = 0.0
    source: str         = "none" # "mediapipe" | "yolo" | "fused"
    knee_lift_l: float  = 0.0   # normalized [0,1] cho walking detect
    knee_lift_r: float  = 0.0


# ── MediaPipe Extractor ────────────────────────────────────────────────────────

class MediaPipeExtractor:
    def __init__(self, model_complexity: int = 1):
        self._mp   = mp.solutions.pose
        self._draw = mp.solutions.drawing_utils
        self.pose  = self._mp.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame_bgr: np.ndarray) -> Tuple[Optional[BodyMetrics], Optional[object], Optional[np.ndarray]]:
        h, w = frame_bgr.shape[:2]
        res  = self.pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None, None, None

        lm = res.pose_landmarks.landmark
        L  = self._mp.PoseLandmark

        def p(idx):
            pt = lm[idx]
            return np.array([pt.x * w, pt.y * h, pt.visibility])

        l_sh = p(L.LEFT_SHOULDER);  r_sh = p(L.RIGHT_SHOULDER)
        l_hp = p(L.LEFT_HIP);       r_hp = p(L.RIGHT_HIP)
        l_kn = p(L.LEFT_KNEE);      r_kn = p(L.RIGHT_KNEE)
        l_an = p(L.LEFT_ANKLE);     r_an = p(L.RIGHT_ANKLE)
        l_wr = p(L.LEFT_WRIST);     r_wr = p(L.RIGHT_WRIST)
        nose = p(L.NOSE)

        shoulder = (l_sh[:2] + r_sh[:2]) / 2
        hip      = (l_hp[:2] + r_hp[:2]) / 2
        ankle    = (l_an[:2] + r_an[:2]) / 2

        spine      = hip - shoulder
        body_angle = math.degrees(math.atan2(abs(spine[0]), abs(spine[1]) + 1e-6))

        pts  = np.array([l_sh[:2], r_sh[:2], l_hp[:2], r_hp[:2],
                         l_kn[:2], r_kn[:2], l_an[:2], r_an[:2]])
        xmin, ymin = pts.min(0);  xmax, ymax = pts.max(0)
        bw = max(xmax - xmin, 1); bh = max(ymax - ymin, 1)

        ankle_hip  = abs(ankle[1] - hip[1]) + 1e-6
        vis        = [l_sh[2], r_sh[2], l_hp[2], r_hp[2], l_an[2], r_an[2]]
        confidence = float(np.mean([v for v in vis if v > 0]) if any(v > 0 for v in vis) else 0.0)

        # Raw normalized keypoints for transformer (33 × 4)
        mp_raw = np.array(
            [[l.x, l.y, l.z, l.visibility] for l in lm],
            dtype=np.float32,
        )

        return BodyMetrics(
            shoulder_y   = shoulder[1],
            hip_y        = hip[1],
            ankle_y      = ankle[1],
            knee_l_y     = l_kn[1],
            knee_r_y     = r_kn[1],
            nose_y       = nose[1],
            wrist_l_y    = l_wr[1],
            wrist_r_y    = r_wr[1],
            body_angle   = body_angle,
            aspect_ratio = bh / bw,
            center_x     = (xmin + xmax) / 2,
            center_y     = (ymin + ymax) / 2,
            bbox_w       = bw,
            bbox_h       = bh,
            confidence   = confidence,
            source       = "mediapipe",
            knee_lift_l  = max(0.0, (hip[1] - l_kn[1])) / ankle_hip,
            knee_lift_r  = max(0.0, (hip[1] - r_kn[1])) / ankle_hip,
        ), res.pose_landmarks, mp_raw

    def draw(self, frame, landmarks, color=(80, 220, 80)):
        lm_spec   = self._draw.DrawingSpec(color=color, thickness=3, circle_radius=5)
        conn_spec = self._draw.DrawingSpec(
            color=tuple(min(c + 40, 255) for c in color), thickness=2)
        self._draw.draw_landmarks(
            frame, landmarks, self._mp.POSE_CONNECTIONS, lm_spec, conn_spec)

    def close(self):
        self.pose.close()


# ── YOLO Pose Extractor ────────────────────────────────────────────────────────

class YOLOPoseExtractor:
    """YOLOv8 Pose — 17 COCO keypoints"""
    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        if not YOLO_AVAILABLE:
            raise ImportError("Chạy: pip install ultralytics")
        self.model = YOLO(model_path)

    def extract(self, frame_bgr: np.ndarray) -> Tuple[Optional[BodyMetrics], Optional[np.ndarray]]:
        results = self.model(frame_bgr, verbose=False)
        if not results or len(results[0].keypoints.xy) == 0:
            return None, None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, None

        idx  = int(boxes.conf.argmax())
        kps  = results[0].keypoints.xy[idx].cpu().numpy()
        conf = (results[0].keypoints.conf[idx].cpu().numpy()
                if results[0].keypoints.conf is not None else np.ones(17))

        def kp(i):
            return np.array([kps[i][0], kps[i][1], float(conf[i])])

        l_sh = kp(5);  r_sh = kp(6)
        l_hp = kp(11); r_hp = kp(12)
        l_kn = kp(13); r_kn = kp(14)
        l_an = kp(15); r_an = kp(16)

        shoulder   = (l_sh[:2] + r_sh[:2]) / 2
        hip        = (l_hp[:2] + r_hp[:2]) / 2
        ankle      = (l_an[:2] + r_an[:2]) / 2
        spine      = hip - shoulder
        body_angle = math.degrees(math.atan2(abs(spine[0]), abs(spine[1]) + 1e-6))

        pts  = np.array([l_sh[:2], r_sh[:2], l_hp[:2], r_hp[:2],
                         l_kn[:2], r_kn[:2], l_an[:2], r_an[:2]])
        xmin, ymin = pts.min(0); xmax, ymax = pts.max(0)
        bw = max(xmax - xmin, 1); bh = max(ymax - ymin, 1)
        ankle_hip  = abs(ankle[1] - hip[1]) + 1e-6

        # Raw normalized keypoints for transformer (17 × 4): [x/w, y/h, 0, conf]
        h, w = frame_bgr.shape[:2]
        yolo_raw = np.zeros((17, 4), dtype=np.float32)
        for i in range(min(17, len(kps))):
            yolo_raw[i] = [kps[i][0] / w, kps[i][1] / h, 0.0, float(conf[i])]

        return BodyMetrics(
            shoulder_y   = shoulder[1],
            hip_y        = hip[1],
            ankle_y      = ankle[1],
            knee_l_y     = l_kn[1],
            knee_r_y     = r_kn[1],
            body_angle   = body_angle,
            aspect_ratio = bh / bw,
            center_x     = (xmin + xmax) / 2,
            center_y     = (ymin + ymax) / 2,
            bbox_w       = bw,
            bbox_h       = bh,
            confidence   = float(boxes.conf[idx]),
            source       = "yolo",
            knee_lift_l  = max(0.0, (hip[1] - l_kn[1])) / ankle_hip,
            knee_lift_r  = max(0.0, (hip[1] - r_kn[1])) / ankle_hip,
        ), yolo_raw


# ── Fusion Engine ──────────────────────────────────────────────────────────────

class PoseEngine:
    def __init__(self, use_yolo: bool = False,
                 yolo_model: str = "yolov8n-pose.pt",
                 model_complexity: int = 1):
        self.mp_ext = MediaPipeExtractor(model_complexity)
        self.yolo_ext: Optional[YOLOPoseExtractor] = None

        if use_yolo and YOLO_AVAILABLE:
            try:
                self.yolo_ext = YOLOPoseExtractor(yolo_model)
            except Exception as e:
                print(f"[PoseEngine] YOLO load failed: {e} → MediaPipe only")
    def process_multi(self, frame_bgr: np.ndarray) -> list[PersonData]:
        """
        Detect nhiều người → trả list PersonData
        Dùng YOLO tracking để gán person_id ổn định
        """
        if self.yolo_ext is None:
            return []
        h, w = frame_bgr.shape[:2]
        results = []

        # YOLO track — gán ID ổn định qua các frame
        yolo_results = self.yolo_ext.model.track(
            frame_bgr,
            persist=True,           # giữ ID qua frame
            tracker="bytetrack.yaml",
            verbose=False,
        )

        if (yolo_results is None or
                yolo_results[0].boxes is None or
                len(yolo_results[0].boxes) == 0):
            return []

        boxes   = yolo_results[0].boxes
        kps_all = yolo_results[0].keypoints

        for i in range(len(boxes)):
            # Lấy person_id từ tracker
            person_id = int(boxes.id[i]) if boxes.id is not None else i

            # YOLO keypoints của người này
            kp_data  = kps_all.data[i].cpu().numpy()  # (17, 3)
            yolo_kp  = np.zeros((17, 4), dtype=np.float32)
            for j, k in enumerate(kp_data[:17]):
                yolo_kp[j] = [k[0]/w, k[1]/h, 0.0, k[2]]

            # Crop frame theo bbox → đưa vào MediaPipe
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame_bgr[y1:y2, x1:x2]

            mp_kp = np.zeros((33, 4), dtype=np.float32)
            mp_metrics = None
            if crop.shape[0] > 10 and crop.shape[1] > 10:
                mp_m, _, mp_raw = self.mp_ext.extract(crop)
                if mp_raw is not None:
                    mp_kp = mp_raw
                if mp_m is not None:
                    mp_metrics = mp_m

            # Fusion raw_kp
            raw_kp = np.concatenate([mp_kp, yolo_kp]).flatten()  # (200,)

            # Tính BodyMetrics từ YOLO keypoints
            metrics = self._yolo_kp_to_metrics(kp_data, boxes.conf[i], h, w)
            if mp_metrics:
                # Weighted fusion nếu có MediaPipe
                wm = float(mp_metrics.confidence)
                wy = float(boxes.conf[i])
                t  = wm + wy + 1e-9
                metrics.center_y     = (mp_metrics.center_y*wm + metrics.center_y*wy)/t
                metrics.body_angle   = (mp_metrics.body_angle*wm + metrics.body_angle*wy)/t
                metrics.aspect_ratio = (mp_metrics.aspect_ratio*wm + metrics.aspect_ratio*wy)/t

            results.append(PersonData(
                person_id = person_id,
                metrics   = metrics,
                raw_kp    = raw_kp,
                bbox      = (x1, y1, x2, y2),
            ))

        return results

    # ── Single-person API ─────────────────────────────────────────────────────

    def _yolo_kp_to_metrics(self, kp_data: np.ndarray, conf: float,
                             h: int, w: int) -> BodyMetrics:
        """Convert YOLO 17-keypoint array (pixel coords) to BodyMetrics."""
        def kp(i):
            if i < len(kp_data):
                row = kp_data[i]
                return np.array([float(row[0]), float(row[1]),
                                 float(row[2]) if len(row) > 2 else 1.0])
            return np.zeros(3)

        l_sh = kp(5);  r_sh = kp(6)
        l_hp = kp(11); r_hp = kp(12)
        l_kn = kp(13); r_kn = kp(14)
        l_an = kp(15); r_an = kp(16)

        shoulder   = (l_sh[:2] + r_sh[:2]) / 2
        hip        = (l_hp[:2] + r_hp[:2]) / 2
        ankle      = (l_an[:2] + r_an[:2]) / 2
        spine      = hip - shoulder
        body_angle = math.degrees(math.atan2(abs(spine[0]), abs(spine[1]) + 1e-6))

        pts  = np.array([l_sh[:2], r_sh[:2], l_hp[:2], r_hp[:2],
                         l_kn[:2], r_kn[:2], l_an[:2], r_an[:2]])
        xmin, ymin = pts.min(0); xmax, ymax = pts.max(0)
        bw = max(xmax - xmin, 1.0); bh = max(ymax - ymin, 1.0)
        ankle_hip = abs(ankle[1] - hip[1]) + 1e-6

        return BodyMetrics(
            shoulder_y   = float(shoulder[1]),
            hip_y        = float(hip[1]),
            ankle_y      = float(ankle[1]),
            knee_l_y     = float(l_kn[1]),
            knee_r_y     = float(r_kn[1]),
            body_angle   = body_angle,
            aspect_ratio = bh / bw,
            center_x     = float((xmin + xmax) / 2),
            center_y     = float((ymin + ymax) / 2),
            bbox_w       = float(bw),
            bbox_h       = float(bh),
            confidence   = float(conf),
            source       = "yolo",
            knee_lift_l  = max(0.0, float(hip[1] - l_kn[1])) / ankle_hip,
            knee_lift_r  = max(0.0, float(hip[1] - r_kn[1])) / ankle_hip,
        )

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[BodyMetrics], Optional[object], np.ndarray]:
        """
        Single-person detection: MediaPipe primary + optional YOLO.
        Returns (metrics_or_None, mp_landmarks_or_None, raw_kp (200,)).
        raw_kp is always a valid (200,) array (zeros when no person detected).
        """
        h, w     = frame_bgr.shape[:2]
        yolo_kp  = np.zeros((17, 4), dtype=np.float32)

        # MediaPipe
        mp_metrics, landmarks, mp_raw = self.mp_ext.extract(frame_bgr)
        mp_kp = mp_raw if mp_raw is not None else np.zeros((33, 4), dtype=np.float32)

        # YOLO (optional, single-frame inference — not tracking)
        if self.yolo_ext is not None:
            try:
                yr = self.yolo_ext.model(frame_bgr, verbose=False)
                if yr[0].keypoints is not None and len(yr[0].keypoints.data) > 0:
                    kp_data = yr[0].keypoints.data[0].cpu().numpy()
                    for i, k in enumerate(kp_data[:17]):
                        yolo_kp[i] = [k[0] / w, k[1] / h, 0.0, k[2]]
                    if mp_metrics is None and yr[0].boxes is not None and len(yr[0].boxes) > 0:
                        mp_metrics = self._yolo_kp_to_metrics(
                            kp_data, float(yr[0].boxes.conf[0]), h, w)
            except Exception:
                pass

        raw_kp = np.concatenate([mp_kp, yolo_kp]).flatten()  # (200,)
        return mp_metrics, landmarks, raw_kp

    def draw_skeleton(
        self, frame: np.ndarray, landmarks, color: tuple = (80, 220, 80)
    ) -> None:
        self.mp_ext.draw(frame, landmarks, color)

    def close(self) -> None:
        self.mp_ext.close()
