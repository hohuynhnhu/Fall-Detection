"""
src/core/overlay.py — Minimal overlay cho camera frame.
Thiết kế tối giản: badge nhỏ, không panel cồng kềnh.
"""
from __future__ import annotations
import cv2
import numpy as np
import math
import time
from collections import deque
from typing import List

from .fall_detector import DetectionResult
from schemas import PoseState

# ── Màu theo trạng thái ────────────────────────────────────────────────────────
STATE_SKELETON_COLORS = {
    PoseState.STANDING: ( 80, 220,  80),
    PoseState.SITTING : ( 80, 180, 255),
    PoseState.LYING   : (255, 160,  30),
    PoseState.WALKING : (180, 100, 255),
    PoseState.FALLING : ( 40,  40, 255),
    PoseState.UNKNOWN : (150, 150, 150),
}
STATE_LABELS_VN = {
    PoseState.STANDING: "DUNG",
    PoseState.SITTING : "NGOI",
    PoseState.LYING   : "NAM",
    PoseState.WALKING : "DI LAI",
    PoseState.FALLING : "TE NGA",
    PoseState.UNKNOWN : "---",
}

_FACE_KNOWN_COLOR   = ( 80, 220,  80)
_FACE_UNKNOWN_COLOR = (160, 160, 160)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _alpha_rect(img, x1, y1, x2, y2, color=(15, 15, 25), alpha=0.65):
    """Hình chữ nhật bán trong suốt."""
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

def _txt(img, text, x, y, scale=0.45, color=(220, 220, 235), thickness=1):
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def _txt_center(img, text, cx, y, scale=0.45, color=(220, 220, 235), thickness=1):
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    _txt(img, text, cx - w // 2, y, scale, color, thickness)

def _text_size(text, scale=0.45, thickness=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return w, h


class OverlayRenderer:
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.alert_t  = 0
        self.vel_graph: deque = deque(maxlen=120)

    # ── 1. State badge — top left, compact ────────────────────────────────────

    def _state_badge(self, frame, result: DetectionResult):
        s     = str(result.state)
        color = STATE_SKELETON_COLORS.get(s, (150, 150, 150))
        label = STATE_LABELS_VN.get(s, "?")

        scale = 0.6
        tw, th = _text_size(label, scale, 2)
        px, py = 14, 14
        pad    = 7
        x2     = px + tw + pad * 2 + 4
        y2     = py + th + pad * 2

        # Nền mờ
        _alpha_rect(frame, px, py, x2, y2, (10, 10, 20), 0.65)
        # Viền màu trái
        cv2.rectangle(frame, (px, py), (px + 3, y2), color, -1)
        # Text
        _txt(frame, label, px + pad + 4, py + th + pad - 1, scale, color, 2)

        # Chỉ báo đi lại (nhỏ, dưới badge)
        if result.is_walking:
            _txt(frame, "~ DI LAI", px + 4, y2 + 14, 0.32, (180, 100, 255))

    # ── 2. Velocity indicator — top right, compact ───────────────────────────

    def _velocity_info(self, frame, result: DetectionResult, threshold: float):
        vy  = result.velocity_y
        self.vel_graph.append(vy)

        ratio = min(abs(vy) / (threshold * 2 + 1e-6), 1.0)
        if ratio > 0.55:
            vc = (60, 60, 255)
        elif ratio > 0.25:
            vc = (50, 200, 255)
        else:
            vc = (80, 200, 80)

        vel_str = f"{abs(vy):.0f} px/s"
        dir_str = "v XUONG" if vy > 5 else ("^ LEN" if vy < -5 else "= YEN")

        tw_v, th_v = _text_size(vel_str, 0.55, 2)
        tw_d, th_d = _text_size(dir_str, 0.30, 1)
        graph_w    = 80
        pad        = 7

        panel_w = max(tw_v, tw_d, graph_w) + pad * 2
        panel_h = th_v + th_d + 28 + pad * 2   # number + direction + mini graph
        px      = self.w - panel_w - 14
        py      = 14

        _alpha_rect(frame, px, py, px + panel_w, py + panel_h, (10, 10, 20), 0.65)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (30, 30, 50), 1)

        # Velocity number
        _txt(frame, vel_str, px + pad, py + th_v + pad, 0.55, vc, 2)
        # Direction
        _txt(frame, dir_str, px + pad, py + th_v + th_d + pad + 6, 0.30, (140, 140, 160))

        # Mini graph
        gy0 = py + panel_h - 14
        bx  = px + pad
        bw  = panel_w - pad * 2
        filled = int(bw * ratio)
        cv2.rectangle(frame, (bx, gy0), (bx + bw, gy0 + 6), (30, 30, 50), -1)
        if filled > 0:
            cv2.rectangle(frame, (bx, gy0), (bx + filled, gy0 + 6), vc, -1)
        # Threshold marker at 50%
        mx = bx + bw // 2
        cv2.line(frame, (mx, gy0 - 2), (mx, gy0 + 8), (255, 200, 50), 1)

    # ── 3. Fall alert — full screen overlay ──────────────────────────────────

    def _fall_alert(self, frame, result: DetectionResult):
        if not result.is_falling:
            self.alert_t = max(0, self.alert_t - 4)
            return
        self.alert_t = min(self.alert_t + 6, 80)

        # Blink border
        alpha = (math.sin(self.alert_t * 0.15) * 0.5 + 0.5) * 0.28
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (self.w, self.h), (0, 0, 160), -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)
        border = int(alpha * 30) + 3
        cv2.rectangle(frame, (0, 0), (self.w - 1, self.h - 1), (50, 50, 255), border)

        # Center card
        cw, ch_ = 360, 90
        bx = self.w // 2 - cw // 2
        by = self.h // 2 - ch_ // 2
        _alpha_rect(frame, bx, by, bx + cw, by + ch_, (10, 8, 8), 0.90)
        cv2.rectangle(frame, (bx, by), (bx + cw, by + ch_), (80, 80, 255), 1)

        blink = (self.alert_t // 10) % 2 == 0
        _txt_center(frame, "PHAT HIEN TE NGA", self.w // 2, by + 32, 0.75,
                    (100, 100, 255) if blink else (200, 100, 100), 2)
        _txt_center(frame, f"Toc do: {result.fall_max_velocity:.0f} px/s",
                    self.w // 2, by + 56, 0.40, (200, 200, 100))
        _txt_center(frame, time.strftime("%H:%M:%S"),
                    self.w // 2, by + 76, 0.35, (140, 140, 160))

    # ── 4. Face boxes — thin border, minimal label ───────────────────────────

    def _face_boxes(self, frame, recognized_persons: List):
        """Chỉ hiện tên khuôn mặt — không vẽ bounding box."""
        for p in recognized_persons:
            if not p.is_known:
                continue
            b     = p.box
            color = _FACE_KNOWN_COLOR

            label = f"{p.name}  {p.confidence:.0%}"
            scale = 0.38
            tw, th = _text_size(label, scale, 1)
            ly = max(b.y1 - 4, th + 4)

            # Label nổi phía trên vị trí mặt, không có box
            _alpha_rect(frame, b.x1, ly - th - 4, b.x1 + tw + 8, ly + 2,
                        (10, 10, 20), 0.72)
            _txt(frame, label, b.x1 + 4, ly - 2, scale, color, 1)

    # ── 5. Bottom status bar — slim ───────────────────────────────────────────

    def _stats_bar(self, frame, result: DetectionResult, fps: float, backend_ok: bool):
        bh = 26
        y0 = self.h - bh
        _alpha_rect(frame, 0, y0, self.w, self.h, (10, 10, 20), 0.80)

        _txt(frame, f"FPS: {fps:.0f}", 10, y0 + 17, 0.40, (100, 100, 130))

        fc = (80, 80, 220) if result.fall_count > 0 else (100, 100, 130)
        _txt(frame, f"Te nga: {result.fall_count} lan",
             100, y0 + 17, 0.40, fc, 2 if result.fall_count > 0 else 1)

        sv = (80, 200, 80) if backend_ok else (180, 80, 80)
        be = "BACKEND: OK" if backend_ok else "BACKEND: OFFLINE"
        tw, _ = _text_size(be, 0.35)
        _txt(frame, be, self.w - tw - 10, y0 + 17, 0.35, sv)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(
            self,
            frame: np.ndarray,
            result: DetectionResult,
            fps: float,
            backend_ok: bool = False,
            fall_vel_threshold: float = 80.0,
            recognized_persons: List = [],
    ) -> np.ndarray:
        self._state_badge(frame, result)
        self._face_boxes(frame, recognized_persons)
        self._stats_bar(frame, result, fps, backend_ok)
        self._fall_alert(frame, result)
        return frame
