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
    metrics: BodyMetrics
    state: str


@dataclass
class DetectionResult:
    state: PoseState            = PoseState.UNKNOWN
    prev_state: PoseState       = PoseState.UNKNOWN
    velocity_y: float           = 0.0   # px/s downward
    velocity_x: float           = 0.0   # px/s horizontal
    is_falling: bool            = False
    is_walking: bool            = False
    fall_count: int             = 0
    metrics: Optional[BodyMetrics] = None
    fall_just_triggered: bool   = False
    fall_max_velocity: float    = 0.0
    fall_start_time: Optional[float] = None


# ── State Classifier ───────────────────────────────────────────────────────────

class StateClassifier:
    def classify(self, m: BodyMetrics, cfg: ThresholdConfig) -> PoseState:
        if m.confidence < 0.50:
            return PoseState.UNKNOWN

        # LYING: spine nearly horizontal OR bbox very wide
        if m.body_angle >= cfg.body_angle_lying or m.aspect_ratio <= cfg.aspect_ratio_lying:
            return PoseState.LYING

        # SITTING: torso short relative to full body height
        body_h  = abs(m.ankle_y - m.shoulder_y) + 1e-6
        torso_h = abs(m.hip_y - m.shoulder_y)
        if (torso_h / body_h) < 0.42:
            return PoseState.SITTING

        return PoseState.STANDING


# ── Velocity Engine ────────────────────────────────────────────────────────────

class VelocityEngine:
    def __init__(self, window: int = 30):
        self._w = window
        self.history: Deque[FrameSnapshot] = deque(maxlen=window)

    def push(self, snap: FrameSnapshot):
        self.history.append(snap)

    def velocity_y(self) -> float:
        """Downward velocity over full window (px/s). Positive = moving down."""
        if len(self.history) < 2:
            return 0.0
        dt = self.history[-1].timestamp - self.history[0].timestamp
        if dt <= 0:
            return 0.0
        return (self.history[-1].metrics.center_y - self.history[0].metrics.center_y) / dt

    def velocity_x(self) -> float:
        """Horizontal speed over full window (px/s)."""
        if len(self.history) < 2:
            return 0.0
        dt = self.history[-1].timestamp - self.history[0].timestamp
        if dt <= 0:
            return 0.0
        return abs(self.history[-1].metrics.center_x - self.history[0].metrics.center_x) / dt

    def frame_velocity_y(self) -> float:
        """Frame-to-frame velocity — more reactive cho fall detection."""
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
                vels.append((snaps[i].metrics.center_y - snaps[i-1].metrics.center_y) / dt)
        return max(vels) if vels else 0.0

    def resize(self, new_window: int):
        if new_window != self._w:
            self._w = new_window
            self.history = deque(list(self.history)[-new_window:], maxlen=new_window)


# ── Walking Detector ───────────────────────────────────────────────────────────

class WalkingDetector:
    """
    Detect đi lại:
    1. vel_x > walk_velocity_threshold
    2. Cả 2 gối xen kẽ nâng trong walk_alternating_window frames
    """
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
        thr = cfg.walk_knee_lift_threshold
        l_lifts = sum(1 for v in list(self.kl)[-n:] if v > thr)
        r_lifts = sum(1 for v in list(self.kr)[-n:] if v > thr)
        return l_lifts >= 2 and r_lifts >= 2

class FallDetector:
    """
    Trigger khi:
    1. Velocity xuống nhanh (> threshold)
    2. History có STANDING/SITTING/WALKING
    3. LYING streak >= fall_confirm_frames
    4. Thời gian từ đứng → nằm <= fall_transition_max_s (loại bỏ nằm chậm)
    5. [NEW] Không phải cúi xuống: hip_y phải xuống cùng shoulder_y
    Recovery: khi trở lại STANDING
    """
    def __init__(self):
        self.state_history: Deque[str]   = deque(maxlen=20)
        self.lying_streak: int           = 0
        self.confirmed: bool             = False
        self.start_time: Optional[float] = None
        self.events: List[dict]          = []
        self._last_upright_t: float      = 0.0
        self._lying_start_t: float       = 0.0
        # Velocity captured at the moment the lying streak begins — used for fast_down check.
        # This avoids false positives from current-frame jitter while the person lies still.
        self._onset_vel: float           = 0.0

        self._onset_aspect_ratio: float = 1.5

        self._baseline_hip_y:      Optional[float] = None
        self._baseline_shoulder_y: Optional[float] = None

        # Settled-lying guard: đếm tổng frame LYING liên tục kể từ lần đứng cuối.
        # Nếu > settle threshold → người đang nằm bình thường, không phải té.
        self._cumulative_lying_frames: int = 0
        self._upright_streak: int          = 0  # frame đứng liên tiếp để reset bộ đếm
        self._first_lying_t: float         = 0.0  # thời điểm đầu tiên vào LYING trong bout này

    def update_baseline(self, metrics: BodyMetrics, state: str):
        """
        [NEW] Cập nhật baseline hip/shoulder khi đang STANDING bình thường.
        Dùng exponential moving average để baseline mượt hơn.
        """
        if state == PoseState.STANDING and metrics.confidence >= 0.5:
            if self._baseline_hip_y is None:
                # Lần đầu tiên — gán thẳng
                self._baseline_hip_y      = metrics.hip_y
                self._baseline_shoulder_y = metrics.shoulder_y
            else:
                # EMA alpha=0.05: cập nhật chậm, tránh bị nhiễu 1 frame
                alpha = 0.05
                self._baseline_hip_y      = (1 - alpha) * self._baseline_hip_y \
                                            + alpha * metrics.hip_y
                self._baseline_shoulder_y = (1 - alpha) * self._baseline_shoulder_y \
                                            + alpha * metrics.shoulder_y

    def _is_bending(self, metrics: BodyMetrics, cfg: ThresholdConfig) -> bool:
        """
        [NEW] Phân biệt cúi xuống vs té ngã dựa vào hip_y và shoulder_y.

        Cúi xuống:
            shoulder_y tăng nhiều (đầu/vai xuống thấp)
            hip_y gần như không đổi (hông vẫn cao)
            → shoulder_drop lớn, hip_drop nhỏ

        Té ngã:
            cả shoulder_y lẫn hip_y đều tăng (toàn thân đổ xuống)
            → cả 2 đều lớn

        Trả về True nếu là cúi xuống → không phải fall.
        """
        if self._baseline_hip_y is None or self._baseline_shoulder_y is None:
            return False  # chưa có baseline → không lọc

        hip_drop      = metrics.hip_y      - self._baseline_hip_y
        shoulder_drop = metrics.shoulder_y - self._baseline_shoulder_y

        # Shoulder xuống đáng kể nhưng hip gần như không đổi → cúi
        is_bend = (
            shoulder_drop > cfg.bend_shoulder_drop_threshold  # vd 40px
            and hip_drop   < cfg.bend_hip_max_drop            # vd 25px
        )
        return is_bend

    def push_state(self, state: str, now: float, onset_vel: float = 0.0,
                   metrics: Optional[BodyMetrics] = None):
        self.state_history.append(state)
        if state == PoseState.LYING:
            self._upright_streak = 0
            self._cumulative_lying_frames += 1
            if self._cumulative_lying_frames == 1:
                self._first_lying_t = now
            if self.lying_streak == 0:
                self._lying_start_t = now
                self._onset_vel = onset_vel
                if metrics is not None:
                    self._onset_aspect_ratio = metrics.aspect_ratio  # lưu tại onset
            self.lying_streak += 1
        else:
            self.lying_streak = 0
            self._last_upright_t = now
            self._upright_streak += 1
            if self._upright_streak >= 15:
                self._cumulative_lying_frames = 0
                self._first_lying_t = 0.0
                self._onset_aspect_ratio = 1.5  # reset khi đứng dậy
        if state == PoseState.STANDING and self.confirmed:
            self.confirmed = False

    def check(self, vel_y: float, max_vel: float,
              cfg: ThresholdConfig, now: float,
              sleep_as_fall: bool = False,
              current_metrics: Optional[BodyMetrics] = None) -> bool:  # [NEW] thêm param
        # Settled-lying guard TRƯỚC — phải chạy trước cả `if self.confirmed`
        # để xóa được confirmed cũ khi người đã nằm bình thường > 1s
        if not sleep_as_fall and self._cumulative_lying_frames > cfg.fall_confirm_frames * 6:
            self.confirmed = False   # xóa false trigger cũ
            return False

        if self.confirmed:
            return False

        # Sleep-as-fall mode: sustained LYING treated as fall (no velocity needed)
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

        recent = list(self.state_history)
        was_up = any(s in (PoseState.STANDING, PoseState.SITTING, PoseState.WALKING)
                     for s in recent[-10:-2])

        fast_down = self._onset_vel > cfg.fall_velocity_threshold
        now_lying = self.lying_streak >= cfg.fall_confirm_frames

        if self._first_lying_t > 0:
            transition_s = now - self._first_lying_t
        else:
            transition_s = self._lying_start_t - self._last_upright_t
        too_slow = (self._first_lying_t > 0 and
                    transition_s > cfg.fall_transition_max_s)

        bending = (
                current_metrics is not None
                and self._is_bending(current_metrics, cfg)
        )

        # Kiểm tra hông có xuống thấp không — phân biệt ngồi→nằm vs té ngã
        # Kiểm tra body có ngả ngang không — phân biệt ngồi→nằm vs té

        # Té thật: aspect_ratio lúc onset còn cao (người đang đứng)
        # Ngồi→nằm: aspect_ratio lúc onset đã thấp sẵn

        # upright_before_fall = self._onset_aspect_ratio > 0.6

        # upright_before_fall = (
        #         self._onset_aspect_ratio > 0.5
        #         and self._onset_vel > 150
        #         and self._onset_aspect_ratio < 1.1
        # )

        upright_before_fall = self._onset_vel > 150

        if was_up and now_lying:
            print(
                f"[FallDetector] body_angle={current_metrics.body_angle:.1f}° | "
                f"aspect_ratio={current_metrics.aspect_ratio:.2f} | "
                f"onset_aspect={self._onset_aspect_ratio:.2f} | "
                f"vel={self._onset_vel:.1f} | transition_s={transition_s:.2f} | "
                f"fast_down={fast_down} | upright_before_fall={upright_before_fall}"
            )

        # if fast_down and was_up and now_lying and not too_slow and not bending and upright_before_fall:
        if fast_down and was_up and now_lying and not bending and upright_before_fall:
            self.confirmed = True
            self.start_time = now
            self.events.append({
                "timestamp": now,
                "velocity": self._onset_vel,
                "transition_s": transition_s,
                "state_before": recent[-10] if len(recent) >= 10 else PoseState.UNKNOWN,
            })
            return True
        return False

    def reset(self):
        self.events.clear()
        self.confirmed        = False
        self.start_time       = None
        self.lying_streak     = 0
        self._last_upright_t  = 0.0
        self._lying_start_t   = 0.0
        self._onset_vel       = 0.0

        self._onset_aspect_ratio = 1.5
        self.state_history.clear()
        self._baseline_hip_y           = None
        self._baseline_shoulder_y      = None
        self._cumulative_lying_frames  = 0
        self._upright_streak           = 0
        self._first_lying_t            = 0.0

    def count(self) -> int:
        return len(self.events)


# ── Detection Pipeline ─────────────────────────────────────────────────────────

class DetectionPipeline:
    def __init__(
        self,
        config:   Optional[ThresholdConfig] = None,
        features: Optional[FeatureConfig]   = None,
    ):
        self.config    = config   or ThresholdConfig()
        self.features  = features or FeatureConfig()
        self.clf       = StateClassifier()
        self.vel       = VelocityEngine(self.config.fall_history_window)
        self.walk      = WalkingDetector()
        self.fall      = FallDetector()
        self.cur_state = PoseState.UNKNOWN
        self.prv_state = PoseState.UNKNOWN
        self.frame_id  = 0


    def update_config(self, cfg: ThresholdConfig):
        self.config = cfg
        self.vel.resize(cfg.fall_history_window)

    def update_features(self, feat: FeatureConfig):
        self.features = feat

    def process(self, metrics: BodyMetrics) -> DetectionResult:
        now = time.time()
        self.frame_id += 1

        state = self.clf.classify(metrics, self.config)

        snap = FrameSnapshot(timestamp=now, metrics=metrics, state=state)
        self.vel.push(snap)

        vel_y     = self.vel.velocity_y()
        vel_x     = self.vel.velocity_x()
        frame_vel = self.vel.frame_velocity_y()
        max_vel   = self.vel.max_velocity_y(last_n=15)

        self.walk.push(metrics)
        if state == PoseState.STANDING and self.walk.is_walking(vel_x, self.config):
            state = PoseState.WALKING

        # [NEW] Cập nhật baseline trước khi push_state
        self.fall.update_baseline(metrics, state)

        self.fall.push_state(state, now, onset_vel=max_vel, metrics=metrics)

        triggered = self.fall.check(
            frame_vel, max_vel, self.config, now,
            sleep_as_fall   = self.features.sleep_as_fall,
            current_metrics = metrics,
        )

        self.prv_state = self.cur_state
        self.cur_state = state

        return DetectionResult(
            state              = state,
            prev_state         = self.prv_state,
            velocity_y         = vel_y,
            velocity_x         = vel_x,
            is_falling         = self.fall.confirmed,
            is_walking         = (state == PoseState.WALKING),
            fall_count         = self.fall.count(),
            metrics            = metrics,
            fall_just_triggered= triggered,
            fall_max_velocity  = max_vel,
            fall_start_time    = self.fall.start_time,
        )

    def reset_falls(self):
        self.fall.reset()
