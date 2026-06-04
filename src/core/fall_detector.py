"""
src/core/fall_detector.py

State Classifier  → STANDING / SITTING / LYING / WALKING / FALLING
Velocity Engine   → sliding window px/s (downward = positive)
Walking Detector  → horizontal velocity + alternating knee lift
Fall Detector     → rule-based trigger + confirm frames
"""
from __future__ import annotations
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Deque

from .pose_engine import BodyMetrics
from schemas import PoseState, ThresholdConfig, FeatureConfig


@dataclass
class FrameSnapshot:
    timestamp: float
    metrics:   BodyMetrics
    state:     str


@dataclass
class DetectionResult:
    state:               PoseState            = PoseState.UNKNOWN
    prev_state:          PoseState            = PoseState.UNKNOWN
    velocity_y:          float                = 0.0
    velocity_x:          float                = 0.0
    is_falling:          bool                 = False
    is_walking:          bool                 = False
    fall_count:          int                  = 0
    metrics:             Optional[BodyMetrics] = None
    fall_just_triggered: bool                 = False
    fall_max_velocity:   float                = 0.0
    fall_start_time:     Optional[float]      = None
    state_changed:       bool                 = False  # True khi stable state vừa đổi


# ── State Stabilizer ───────────────────────────────────────────────────────────

class StateStabilizer:
    """
    Cần N frame liên tiếp cùng raw_state mới confirm lên stable_state.
    LYING thấp (3) để fall detector nhận sớm.
    SITTING cao (10) vì camera góc cao dễ nhầm với STANDING.
    UNKNOWN cao (10) để không về UNKNOWN khi mất detection 1-2 frame.
    """
    CONFIRM = {
        PoseState.LYING: 4,
        PoseState.SITTING: 5,
        PoseState.WALKING: 5,
        PoseState.STANDING: 6,
        PoseState.UNKNOWN: 10,
    }
    def __init__(self):
        self.stable_state: PoseState = PoseState.UNKNOWN
        self._candidate:   PoseState = PoseState.UNKNOWN
        self._streak:      int       = 0

    def update(self, raw_state: PoseState) -> PoseState:
        if raw_state == self._candidate:
            self._streak += 1
        else:
            self._candidate = raw_state
            self._streak    = 1
        needed = self.CONFIRM.get(self._candidate, 4)
        if self._streak >= needed:
            self.stable_state = self._candidate
        return self.stable_state

    def reset(self):
        self.stable_state = PoseState.UNKNOWN
        self._candidate   = PoseState.UNKNOWN
        self._streak      = 0


# ── State Classifier ───────────────────────────────────────────────────────────

class StateClassifier:
    """
    Phân loại tư thế thuần từ YOLO keypoints — không dùng z_diff.

    Feature chính:
      body_angle     — góc vector shoulder→hip so với trục dọc (0°=đứng, 90°=nằm)
      aspect_ratio   — H/W của bounding box 8 keypoints
      hip_ankle_ratio — (ankle_y - hip_y) / bbox_h
                        đứng: 0.45-0.70 | ngồi: 0.15-0.45 | nằm: <0.20 hoặc âm

    Điều chỉnh cho camera góc cao (tủ lạnh):
      - Người nằm nhìn từ cao: bbox không bẹt → KHÔNG dùng aspect_ratio cho LYING
      - LYING chỉ dựa vào body_angle + hip_ankle_ratio
      - hip_ankle_ratio âm (mắt cá cao hơn hông) = chắc chắn nằm/ngã
    """

    def classify(self, m: BodyMetrics, cfg: ThresholdConfig,
                 sitting_prob: float = 0.0) -> PoseState:

        # Reject detection yếu
        if m.confidence < cfg.min_confidence:
            return PoseState.UNKNOWN

        # Guard: bbox quá nhỏ
        if m.bbox_h < 100:
            return PoseState.UNKNOWN

        hip_ankle_ratio = (
            (m.ankle_y - m.hip_y) / m.bbox_h
            if m.bbox_h > 1e-3 else 1.0
        )

        # ── LYING ─────────────────────────────────────────────────────────────
        if not m.ankle_reliable:
            if m.body_angle >= cfg.body_angle_lying:
                return PoseState.LYING
            return PoseState.STANDING

        # Case 1: mắt cá cao hơn hông → chắc chắn nằm/ngã
        if hip_ankle_ratio < 0:
            return PoseState.LYING

        # Case 2a: hip_ankle rất thấp
        if hip_ankle_ratio < 0.20:
            return PoseState.LYING

        # Case 2b: góc thân >= 50° VÀ hip_ankle < 0.56
        if m.body_angle >= 50.0 and hip_ankle_ratio < 0.56:
            return PoseState.LYING

        # Case 2c: góc thân >= 30° VÀ hip_ankle < 0.35
        if m.body_angle >= 30.0 and hip_ankle_ratio < 0.35:
            return PoseState.LYING

            # ── SITTING ───────────────────────────────────────────────────────────
        if (cfg.sitting_ar_min <= m.aspect_ratio <= cfg.sitting_ar_max
            and m.body_angle < cfg.sitting_angle_max
            and 0.10 <= hip_ankle_ratio < cfg.sitting_hip_ankle_max):
            return PoseState.SITTING


        # ── STANDING (mặc định) ───────────────────────────────────────────────
        return PoseState.STANDING

# ── Velocity Engine ────────────────────────────────────────────────────────────

class VelocityEngine:
    def __init__(self, window: int = 30):
        self._w = window
        self.history: Deque[FrameSnapshot] = deque(maxlen=window)

    def push(self, snap: FrameSnapshot):
        self.history.append(snap)

    def velocity_y(self) -> float:
        if len(self.history) < 2:
            return 0.0
        dt = self.history[-1].timestamp - self.history[0].timestamp
        if dt <= 0:
            return 0.0
        return (self.history[-1].metrics.center_y - self.history[0].metrics.center_y) / dt

    def velocity_x(self) -> float:
        if len(self.history) < 2:
            return 0.0
        dt = self.history[-1].timestamp - self.history[0].timestamp
        if dt <= 0:
            return 0.0
        return abs(self.history[-1].metrics.center_x - self.history[0].metrics.center_x) / dt

    def frame_velocity_y(self) -> float:
        if len(self.history) < 2:
            return 0.0
        a, b = self.history[-2], self.history[-1]
        dt = b.timestamp - a.timestamp
        return (b.metrics.center_y - a.metrics.center_y) / dt if dt > 0 else 0.0

    def max_velocity_y(self, last_n: int = 10) -> float:
        snaps = list(self.history)[-last_n:]
        vels  = []
        for i in range(1, len(snaps)):
            dt = snaps[i].timestamp - snaps[i-1].timestamp
            if dt > 0:
                vels.append(
                    (snaps[i].metrics.center_y - snaps[i-1].metrics.center_y) / dt
                )
        return max(vels) if vels else 0.0

    def resize(self, new_window: int):
        if new_window != self._w:
            self._w = new_window
            self.history = deque(list(self.history)[-new_window:], maxlen=new_window)


# ── Walking Detector ───────────────────────────────────────────────────────────

class WalkingDetector:
    def __init__(self):
        self.kl: Deque[float] = deque(maxlen=30)
        self.kr: Deque[float] = deque(maxlen=30)

    def push(self, m: BodyMetrics):
        self.kl.append(m.knee_lift_l)
        self.kr.append(m.knee_lift_r)

    def is_walking(self, vel_x: float, cfg: ThresholdConfig) -> bool:
        if vel_x < cfg.walk_velocity_threshold:
            return False
        n = cfg.walk_alternating_window
        if len(self.kl) < n:
            return False
        thr     = cfg.walk_knee_lift_threshold
        l_lifts = sum(1 for v in list(self.kl)[-n:] if v > thr)
        r_lifts = sum(1 for v in list(self.kr)[-n:] if v > thr)
        return l_lifts >= 2 and r_lifts >= 2


# ── Fall Detector ──────────────────────────────────────────────────────────────

class FallDetector:
    """
    Trigger khi:
    1. Velocity đủ nhanh (onset_vel > threshold)
    2. History gần đây có STANDING/SITTING/WALKING
    3. LYING streak >= fall_confirm_frames
    4. Không phải cúi người (bending guard)
    5. Velocity sanity: loại bỏ velocity vật lý bất khả thi (> 5000 px/s)
    6. Warmup guard: bỏ qua 45 frame đầu khi person mới detect

    Recovery: khi trở lại STANDING
    """

    def __init__(self):
        self.state_history: Deque[str]    = deque(maxlen=20)
        self.lying_streak:  int           = 0
        self.confirmed:     bool          = False
        self.start_time:    Optional[float] = None
        self.events:        List[dict]    = []
        self._last_upright_t: float       = 0.0
        self._lying_start_t:  float       = 0.0
        self._onset_vel:      float       = 0.0
        self._onset_aspect_ratio: float   = 1.5
        self._baseline_hip_y:      Optional[float] = None
        self._baseline_shoulder_y: Optional[float] = None
        self._cumulative_lying_frames: int = 0
        self._upright_streak:          int = 0
        self._first_lying_t:         float = 0.0
        # [NEW] warmup: bỏ qua N frame đầu khi person vừa được detect
        self._stable_frame_count: int     = 0

    def update_baseline(self, metrics: BodyMetrics, state: str):
        if state == PoseState.STANDING and metrics.confidence >= 0.5:
            if self._baseline_hip_y is None:
                self._baseline_hip_y      = metrics.hip_y
                self._baseline_shoulder_y = metrics.shoulder_y
            else:
                alpha = 0.05
                self._baseline_hip_y = (
                    (1 - alpha) * self._baseline_hip_y + alpha * metrics.hip_y
                )
                self._baseline_shoulder_y = (
                    (1 - alpha) * self._baseline_shoulder_y + alpha * metrics.shoulder_y
                )

    def _is_bending(self, metrics: BodyMetrics, cfg: ThresholdConfig) -> bool:
        if self._baseline_hip_y is None or self._baseline_shoulder_y is None:
            return False
        hip_drop      = metrics.hip_y      - self._baseline_hip_y
        shoulder_drop = metrics.shoulder_y - self._baseline_shoulder_y
        return (
            shoulder_drop > cfg.bend_shoulder_drop_threshold
            and hip_drop  < cfg.bend_hip_max_drop
        )

    def push_state(self, state: str, now: float, onset_vel: float = 0.0,
                   metrics: Optional[BodyMetrics] = None):
        # [NEW] đếm frame để warmup guard
        self._stable_frame_count += 1

        self.state_history.append(state)
        if state == PoseState.LYING:
            self._upright_streak = 0
            self._cumulative_lying_frames += 1
            if self._cumulative_lying_frames == 1:
                self._first_lying_t = now
            if self.lying_streak == 0:
                self._lying_start_t      = now
                self._onset_vel          = onset_vel
                if metrics is not None:
                    self._onset_aspect_ratio = metrics.aspect_ratio
            self.lying_streak += 1
        else:
            self.lying_streak    = 0
            self._last_upright_t = now
            self._upright_streak += 1
            if self._upright_streak >= 15:
                self._cumulative_lying_frames = 0
                self._first_lying_t           = 0.0
                self._onset_aspect_ratio      = 1.5
        if state == PoseState.STANDING and self.confirmed:
            self.confirmed = False

    def check(self, vel_y: float, max_vel: float,
              cfg: ThresholdConfig, now: float,
              sleep_as_fall: bool = False,
              current_metrics: Optional[BodyMetrics] = None) -> bool:

        # [NEW] Warmup guard: bỏ qua 45 frame đầu (~1.5s @ 30fps)
        # Tránh false positive khi tracker vừa detect, velocity chưa ổn định
        if self._stable_frame_count < 20:
            return False

        # [NEW] Velocity sanity: loại bỏ velocity vật lý bất khả thi
        # (xảy ra khi person_id bị reset, center_y nhảy đột ngột)
        if self._onset_vel > 5000:
            return False

        # Settled-lying guard: người đã nằm bình thường lâu → không phải té
        if not sleep_as_fall and self._cumulative_lying_frames > cfg.fall_confirm_frames * 6:
            self.confirmed = False
            return False

        if self.confirmed:
            return False

        # Sleep-as-fall mode
        if sleep_as_fall and self.lying_streak >= cfg.sleep_confirm_frames:
            self.confirmed  = True
            self.start_time = now
            recent = list(self.state_history)
            self.events.append({
                "timestamp"   : now,
                "velocity"    : 0.0,
                "transition_s": 0.0,
                "state_before": (recent[-cfg.sleep_confirm_frames - 1]
                                 if len(recent) > cfg.sleep_confirm_frames
                                 else PoseState.UNKNOWN),
            })
            return True

        recent    = list(self.state_history)
        was_up = any(s in (PoseState.STANDING, PoseState.WALKING)
                     for s in recent[-10:-2])
        fast_down = self._onset_vel > cfg.fall_velocity_threshold
        now_lying = self.lying_streak >= cfg.fall_confirm_frames

        if self._first_lying_t > 0:
            transition_s = now - self._first_lying_t
        else:
            transition_s = self._lying_start_t - self._last_upright_t

        bending             = (current_metrics is not None
                               and self._is_bending(current_metrics, cfg))
        upright_before_fall = self._onset_vel > 150

        if was_up and now_lying:
            print(
                f"[FallDetector] body_angle={current_metrics.body_angle:.1f}° | "
                f"aspect_ratio={current_metrics.aspect_ratio:.2f} | "
                f"onset_aspect={self._onset_aspect_ratio:.2f} | "
                f"vel={self._onset_vel:.1f} | transition_s={transition_s:.2f} | "
                f"fast_down={fast_down} | upright_before_fall={upright_before_fall}"
            )

        if fast_down and was_up and now_lying and not bending and upright_before_fall:
            self.confirmed  = True
            self.start_time = now
            self.events.append({
                "timestamp"   : now,
                "velocity"    : self._onset_vel,
                "transition_s": transition_s,
                "state_before": recent[-10] if len(recent) >= 10 else PoseState.UNKNOWN,
            })
            return True
        return False

    def reset(self):
        self.events.clear()
        self.confirmed                = False
        self.start_time               = None
        self.lying_streak             = 0
        self._last_upright_t          = 0.0
        self._lying_start_t           = 0.0
        self._onset_vel               = 0.0
        self._onset_aspect_ratio      = 1.5
        self.state_history.clear()
        self._baseline_hip_y          = None
        self._baseline_shoulder_y     = None
        self._cumulative_lying_frames = 0
        self._upright_streak          = 0
        self._first_lying_t           = 0.0
        # self._stable_frame_count      = 0  # [NEW]

    def count(self) -> int:
        return len(self.events)


# ── Detection Pipeline ─────────────────────────────────────────────────────────

class DetectionPipeline:
    def __init__(
        self,
        config:   Optional[ThresholdConfig] = None,
        features: Optional[FeatureConfig]   = None,
    ):
        self.config     = config   or ThresholdConfig()
        self.features   = features or FeatureConfig()
        self.clf        = StateClassifier()
        self.vel        = VelocityEngine(self.config.fall_history_window)
        self.walk       = WalkingDetector()
        self.fall       = FallDetector()
        self.stabilizer = StateStabilizer()
        self.cur_state  = PoseState.UNKNOWN
        self.prv_state  = PoseState.UNKNOWN
        self.frame_id   = 0
        self._last_reported: PoseState = PoseState.UNKNOWN
        self.last_result = DetectionResult(state=PoseState.UNKNOWN)

    def update_config(self, cfg: ThresholdConfig):
        self.config = cfg
        self.vel.resize(cfg.fall_history_window)

    def update_features(self, feat: FeatureConfig):
        self.features = feat

    def _log_state_change(self, new_state: PoseState, m: BodyMetrics):
        """In log chỉ khi state thay đổi, kèm các feature để debug."""
        hip_ankle = (
            (m.ankle_y - m.hip_y) / m.bbox_h
            if m.bbox_h > 1e-3 else 0.0
        )
        if new_state == PoseState.LYING:
            print(f"[Pose] LYING    | angle={m.body_angle:.1f}°"
                  f" ar={m.aspect_ratio:.2f}"
                  f" hip_ankle={hip_ankle:.2f}"
                  f" conf={m.confidence:.2f}")

        elif new_state == PoseState.STANDING:
            print(f"[Pose] STANDING | angle={m.body_angle:.1f}°"
                  f" ar={m.aspect_ratio:.2f}"
                  f" hip_ankle={hip_ankle:.2f}"
                  f" conf={m.confidence:.2f}")
        elif new_state == PoseState.WALKING:
            print(f"[Pose] WALKING  | angle={m.body_angle:.1f}°"
                  f" ar={m.aspect_ratio:.2f}"
                  f" conf={m.confidence:.2f}")
        elif new_state == PoseState.UNKNOWN:
            print(f"[Pose] UNKNOWN  | conf={m.confidence:.2f}")

    def process(self, metrics: BodyMetrics,
                sitting_prob: float = 0.0) -> DetectionResult:
        now = time.time()
        self.frame_id += 1

        raw_state = self.clf.classify(metrics, self.config, sitting_prob=sitting_prob)

        snap = FrameSnapshot(timestamp=now, metrics=metrics, state=raw_state)
        self.vel.push(snap)

        vel_y     = self.vel.velocity_y()
        vel_x     = self.vel.velocity_x()
        frame_vel = self.vel.frame_velocity_y()
        max_vel   = self.vel.max_velocity_y(last_n=15)

        self.walk.push(metrics)
        if raw_state == PoseState.STANDING and self.walk.is_walking(vel_x, self.config):
            raw_state = PoseState.WALKING

        # Fall detector dùng raw_state (không qua stabilizer)
        # để lying_streak tăng ngay, không bị delay
        self.fall.update_baseline(metrics, raw_state)
        self.fall.push_state(raw_state, now, onset_vel=max_vel, metrics=metrics)
        triggered = self.fall.check(
            frame_vel, max_vel, self.config, now,
            sleep_as_fall=self.features.sleep_as_fall,
            current_metrics=metrics,
        )

        # Stabilizer chỉ cho UI display
        stable_state = self.stabilizer.update(raw_state)

        self.prv_state = self.cur_state
        self.cur_state = stable_state

        # Log + state_changed chỉ khi stable state thực sự đổi
        state_changed = (stable_state != self._last_reported)
        if state_changed:
            self._log_state_change(stable_state, metrics)
            self._last_reported = stable_state

        return DetectionResult(
            state               = stable_state,
            prev_state          = self.prv_state,
            velocity_y          = vel_y,
            velocity_x          = vel_x,
            is_falling          = self.fall.confirmed,
            is_walking          = (stable_state == PoseState.WALKING),
            fall_count          = self.fall.count(),
            metrics             = metrics,
            fall_just_triggered = triggered,
            fall_max_velocity   = max_vel,
            fall_start_time     = self.fall.start_time,
            state_changed       = state_changed,
        )

    def reset_falls(self):
        self.fall.reset()
        self.stabilizer.reset()
        self._last_reported = PoseState.UNKNOWN
