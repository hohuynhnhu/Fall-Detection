"""
src/core/pose_engine.py

YOLOv8 Pose primary — MediaPipe chỉ dùng để vẽ skeleton (optional)
- YOLO ByteTrack: tracking ID ổn định, không hallucinate đồ vật
- MediaPipe: KHÔNG dùng để detect/classify nữa, chỉ vẽ keypoints nếu cần
"""
from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
# Đường dẫn tuyệt đối đến bytetrack_custom.yaml
_TRACKER_CFG = str(Path(__file__).resolve().parent.parent / "bytetrack_custom.yaml")
#                                          ↑ chỉ 2 parent: core → src

@dataclass
class PersonData:
    person_id: int
    metrics:   "BodyMetrics"
    raw_kp:    np.ndarray   # (200,) — [mp_zeros(33×4) + yolo(17×4)]
    bbox:      tuple        # (x1, y1, x2, y2)


@dataclass
class BodyMetrics:
    shoulder_y:  float = 0.0
    hip_y:       float = 0.0
    ankle_y:     float = 0.0
    knee_l_y:    float = 0.0
    knee_r_y:    float = 0.0
    nose_y:      float = 0.0
    wrist_l_y:   float = 0.0
    wrist_r_y:   float = 0.0
    body_angle:  float = 0.0   # 0°=đứng, 90°=nằm
    aspect_ratio: float = 1.5  # H/W bbox keypoints
    center_x:    float = 0.0
    center_y:    float = 0.0
    bbox_w:      float = 0.0
    bbox_h:      float = 0.0
    confidence:  float = 0.0
    source:      str   = "yolo"
    knee_lift_l: float = 0.0
    knee_lift_r: float = 0.0
    hip_z:       float = 0.0
    ankle_z:     float = 0.0
    ankle_reliable: bool = True


# ── YOLO Pose Extractor ────────────────────────────────────────────────────────

class YOLOPoseExtractor:
    """YOLOv8 Pose — 17 COCO keypoints + ByteTrack tracking"""

    # COCO keypoint indices
    _NOSE        = 0
    _L_SHOULDER  = 5;  _R_SHOULDER = 6
    _L_HIP       = 11; _R_HIP      = 12
    _L_KNEE      = 13; _R_KNEE     = 14
    _L_ANKLE     = 15; _R_ANKLE    = 16
    _L_WRIST     = 9;  _R_WRIST    = 10

    # Confidence tối thiểu của keypoint để dùng
    _MIN_KP_CONF = 0.50

    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        if not YOLO_AVAILABLE:
            raise ImportError("pip install ultralytics")
        self.model = YOLO(model_path)

    def _kp(self, kp_data: np.ndarray, idx: int) -> np.ndarray:
        """Trả [x, y, conf]. Nếu conf thấp → trả zeros."""
        if idx >= len(kp_data):
            return np.zeros(3)
        row = kp_data[idx]
        x, y = float(row[0]), float(row[1])
        c    = float(row[2]) if len(row) > 2 else 1.0
        return np.array([x, y, c])

    def _safe_mid(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Trung điểm 2 keypoint, ưu tiên bên nào có conf cao hơn."""
        if a[2] >= self._MIN_KP_CONF and b[2] >= self._MIN_KP_CONF:
            return (a[:2] + b[:2]) / 2
        elif a[2] >= self._MIN_KP_CONF:
            return a[:2]
        elif b[2] >= self._MIN_KP_CONF:
            return b[:2]
        return (a[:2] + b[:2]) / 2  # cả 2 đều thấp → dùng trung bình

    def kp_to_metrics(self, kp_data: np.ndarray, bbox_conf: float,
                      frame_h: int, frame_w: int) -> BodyMetrics:
        """Convert 17 YOLO keypoints → BodyMetrics."""
        kp = self._kp

        l_sh = kp(kp_data, self._L_SHOULDER)
        r_sh = kp(kp_data, self._R_SHOULDER)
        l_hp = kp(kp_data, self._L_HIP)
        r_hp = kp(kp_data, self._R_HIP)
        l_kn = kp(kp_data, self._L_KNEE)
        r_kn = kp(kp_data, self._R_KNEE)
        l_an = kp(kp_data, self._L_ANKLE)
        r_an = kp(kp_data, self._R_ANKLE)
        l_wr = kp(kp_data, self._L_WRIST)
        r_wr = kp(kp_data, self._R_WRIST)
        nose = kp(kp_data, self._NOSE)

        shoulder = self._safe_mid(l_sh, r_sh)
        hip      = self._safe_mid(l_hp, r_hp)
        ankle    = self._safe_mid(l_an, r_an)

        # body_angl e: góc vector shoulder→hip so với trục dọc
        spine      = hip - shoulder
        body_angle = math.degrees(
            math.atan2(abs(spine[0]), abs(spine[1]) + 1e-6)
        )
        ankle_reliable = (l_an[2] >= self._MIN_KP_CONF or r_an[2] >= self._MIN_KP_CONF)


        # Bounding box từ 8 keypoints chính
        pts = np.array([
            l_sh[:2], r_sh[:2],
            l_hp[:2], r_hp[:2],
            l_kn[:2], r_kn[:2],
            l_an[:2], r_an[:2],
        ])
        xmin, ymin = pts.min(0)
        xmax, ymax = pts.max(0)
        bw = max(xmax - xmin, 1.0)
        bh = max(ymax - ymin, 1.0)

        ankle_hip = abs(float(ankle[1]) - float(hip[1])) + 1e-6

        # Confidence = trung bình conf của 6 keypoint cốt lõi
        core_confs = [l_sh[2], r_sh[2], l_hp[2], r_hp[2], l_an[2], r_an[2]]
        confidence = float(np.mean(core_confs))

        return BodyMetrics(
            shoulder_y   = float(shoulder[1]),
            hip_y        = float(hip[1]),
            ankle_y      = float(ankle[1]),
            knee_l_y     = float(l_kn[1]),
            knee_r_y     = float(r_kn[1]),
            nose_y       = float(nose[1]),
            wrist_l_y    = float(l_wr[1]),
            wrist_r_y    = float(r_wr[1]),
            body_angle   = body_angle,
            aspect_ratio = bh / bw,
            center_x     = float((xmin + xmax) / 2),
            center_y     = float((ymin + ymax) / 2),
            bbox_w       = float(bw),
            bbox_h       = float(bh),
            confidence   = confidence,
            source       = "yolo",
            knee_lift_l  = max(0.0, float(hip[1] - l_kn[1])) / ankle_hip,
            knee_lift_r  = max(0.0, float(hip[1] - r_kn[1])) / ankle_hip,
            hip_z        = 0.0,
            ankle_z      = 0.0,
            ankle_reliable=ankle_reliable,
        )


# ── MediaPipe (chỉ dùng để vẽ skeleton) ───────────────────────────────────────

class MediaPipeDrawer:
    """Chỉ dùng để vẽ skeleton lên frame — KHÔNG dùng để detect/classify."""
    def __init__(self, model_complexity: int = 1):
        self._mp   = mp.solutions.pose
        self._draw = mp.solutions.drawing_utils
        self.pose  = self._mp.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

    def extract_landmarks(self, frame_bgr: np.ndarray):
        """Trả landmarks để vẽ, hoặc None nếu không detect được."""
        res = self.pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return res.pose_landmarks if res.pose_landmarks else None

    def draw(self, frame: np.ndarray, landmarks, color=(80, 220, 80)):
        lm_spec = self._draw.DrawingSpec(color=color, thickness=-1, circle_radius=4)
        self._draw.draw_landmarks(frame, landmarks, None, lm_spec, lm_spec)

    def close(self):
        self.pose.close()


# ── PoseEngine ─────────────────────────────────────────────────────────────────

class PoseEngine:
    """
    YOLO làm primary detector + tracker.
    MediaPipe chỉ dùng để vẽ skeleton (tuỳ chọn, không ảnh hưởng logic).
    """

    def __init__(self, use_yolo: bool = True,
                 yolo_model: str = "yolov8n-pose.pt",
                 model_complexity: int = 1):

        self.yolo_ext: Optional[YOLOPoseExtractor] = None
        self.mp_drawer: Optional[MediaPipeDrawer]  = None

        if YOLO_AVAILABLE:
            try:
                self.yolo_ext = YOLOPoseExtractor(yolo_model)
                print("[PoseEngine] YOLO Pose loaded OK")
            except Exception as e:
                print(f"[PoseEngine] YOLO load failed: {e}")
        else:
            print("[PoseEngine] ultralytics không có — không thể detect")

        # MediaPipe chỉ để vẽ
        try:
            self.mp_drawer = MediaPipeDrawer(model_complexity)
        except Exception:
            self.mp_drawer = None

    # ── Multi-person (YOLO tracking) ──────────────────────────────────────────

    def process_multi(self, frame_bgr: np.ndarray) -> List[PersonData]:
        """
        Detect + track nhiều người bằng YOLO ByteTrack.
        Trả list PersonData — mỗi người có ID ổn định, metrics từ YOLO thuần.
        """
        if self.yolo_ext is None:
            return []

        h, w = frame_bgr.shape[:2]

        yolo_results = self.yolo_ext.model.track(
            frame_bgr,
            persist=True,
            tracker=_TRACKER_CFG,
            verbose=False,
        )

        if (yolo_results is None
                or yolo_results[0].boxes is None
                or len(yolo_results[0].boxes) == 0):
            return []

        boxes    = yolo_results[0].boxes
        kps_all  = yolo_results[0].keypoints
        results  = []

        for i in range(len(boxes)):
            bbox_conf = float(boxes.conf[i])

            # Bỏ qua detection yếu
            if bbox_conf < 0.45:
                continue

            person_id = int(boxes.id[i]) if boxes.id is not None else i
            kp_data   = kps_all.data[i].cpu().numpy()   # (17, 3): x, y, conf

            # Raw keypoints cho Transformer
            yolo_kp = np.zeros((17, 4), dtype=np.float32)
            for j, k in enumerate(kp_data[:17]):
                yolo_kp[j] = [k[0]/w, k[1]/h, 0.0, float(k[2])]

            # Phần MediaPipe để trống (zeros) — Transformer đã được train với format này
            mp_kp  = np.zeros((33, 4), dtype=np.float32)
            raw_kp = np.concatenate([mp_kp, yolo_kp]).flatten()  # (200,)

            # Bbox từ YOLO detector
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)

            # Metrics thuần YOLO — không fusion với MediaPipe
            metrics = self.yolo_ext.kp_to_metrics(kp_data, bbox_conf, h, w)

            results.append(PersonData(
                person_id = person_id,
                metrics   = metrics,
                raw_kp    = raw_kp,
                bbox      = (x1, y1, x2, y2),
            ))

        return results

    # ── Single-person fallback (YOLO only, không tracking) ────────────────────

    def process(self, frame_bgr: np.ndarray
                ) -> Tuple[Optional[BodyMetrics], Optional[object], np.ndarray]:
        """
        Single-frame YOLO inference — không tracking, không MediaPipe.
        Chỉ dùng khi process_multi() trả về rỗng.
        Trả (metrics_or_None, None, raw_kp(200,)).
        """
        h, w   = frame_bgr.shape[:2]
        raw_kp = np.zeros(200, dtype=np.float32)

        if self.yolo_ext is None:
            return None, None, raw_kp

        try:
            yr = self.yolo_ext.model(frame_bgr, verbose=False)
            if (yr is None
                    or yr[0].boxes is None
                    or len(yr[0].boxes) == 0
                    or yr[0].keypoints is None):
                return None, None, raw_kp

            # Lấy người có confidence cao nhất
            best = int(yr[0].boxes.conf.argmax())
            if float(yr[0].boxes.conf[best]) < 0.45:
                return None, None, raw_kp

            kp_data = yr[0].keypoints.data[best].cpu().numpy()
            yolo_kp = np.zeros((17, 4), dtype=np.float32)
            for j, k in enumerate(kp_data[:17]):
                yolo_kp[j] = [k[0]/w, k[1]/h, 0.0, float(k[2])]

            mp_kp  = np.zeros((33, 4), dtype=np.float32)
            raw_kp = np.concatenate([mp_kp, yolo_kp]).flatten()

            metrics = self.yolo_ext.kp_to_metrics(
                kp_data, float(yr[0].boxes.conf[best]), h, w
            )
            return metrics, None, raw_kp

        except Exception:
            return None, None, raw_kp

    # ── Skeleton drawing (MediaPipe drawer) ───────────────────────────────────

    def draw_skeleton(self, frame: np.ndarray, landmarks,
                      color: tuple = (80, 220, 80)) -> None:
        """Vẽ skeleton nếu có landmarks từ MediaPipe drawer."""
        if self.mp_drawer is not None and landmarks is not None:
            self.mp_drawer.draw(frame, landmarks, color)

    def get_mp_landmarks(self, frame_bgr: np.ndarray):
        """Lấy landmarks từ MediaPipe để vẽ (không ảnh hưởng detection)."""
        if self.mp_drawer is None:
            return None
        return self.mp_drawer.extract_landmarks(frame_bgr)

    def close(self) -> None:
        if self.mp_drawer is not None:
            self.mp_drawer.close()

