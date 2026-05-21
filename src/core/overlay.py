"""
src/core/overlay.py — Vẽ overlay lên camera frame
"""
from __future__ import annotations
import cv2
import numpy as np
import math
import time
from collections import deque
from typing import Optional, List

from .fall_detector import DetectionResult
from schemas import PoseState

STATE_SKELETON_COLORS = {
    PoseState.STANDING: (80,  220,  80),
    PoseState.SITTING : (80,  180, 255),
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
_C = {"panel": (20,20,35), "text": (220,220,235), "sub": (120,120,145), "accent": (255,210,50)}

_FACE_KNOWN_COLOR   = (80,  220, 80)    # xanh lá — thành viên nhận ra
_FACE_UNKNOWN_COLOR = (80,  160, 255)   # xanh lam — không nhận ra


def _rrect(img, x1, y1, x2, y2, r, color, alpha=0.82):
    ov = img.copy()
    cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(ov, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(ov, (cx, cy), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

def _txt(img, t, x, y, s=0.55, color=(220,220,235), th=1):
    cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, color, th, cv2.LINE_AA)

def _txt_c(img, t, cx, y, s=0.55, color=(220,220,235), th=1):
    (w, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, th)
    _txt(img, t, cx - w//2, y, s, color, th)


class OverlayRenderer:
    def __init__(self, w: int, h: int):
        self.w = w; self.h = h
        self.alert_t = 0
        self.vel_graph: deque = deque(maxlen=80)

    def _state_badge(self, frame, result: DetectionResult):
        s = str(result.state)
        color = STATE_SKELETON_COLORS.get(s, (150, 150, 150))
        label = STATE_LABELS_VN.get(s, "?")
        _rrect(frame, 12, 12, 215, 72, 10, _C["panel"])
        cv2.rectangle(frame, (12, 12), (18, 72), color, -1)
        _txt(frame, "TRANG THAI", 24, 30, 0.36, _C["sub"])
        _txt(frame, label, 24, 58, 0.88, color, 2)
        if result.is_walking:
            _txt(frame, "~ DANG DI LAI", 24, 80, 0.36, (180, 100, 255))

    def _velocity_panel(self, frame, result: DetectionResult, threshold: float):
        vy = result.velocity_y
        self.vel_graph.append(vy)
        px = self.w - 185
        _rrect(frame, px, 12, px+173, 150, 10, _C["panel"])
        _txt(frame, "VAN TOC TE", px+10, 30, 0.36, _C["sub"])

        ratio = min(abs(vy) / (threshold*2 + 1e-6), 1.0)
        bc = (int(ratio*175+80), int((1-ratio)*200+30), 40) if ratio > 0.3 else (80, 220, 80)
        _txt(frame, f"{abs(vy):.0f} px/s", px+10, 60, 0.8, bc, 2)

        if vy > 5:    _txt(frame, "v XUONG",   px+10, 78, 0.36, (255, 100, 100))
        elif vy < -5: _txt(frame, "^ LEN",     px+10, 78, 0.36, (100, 255, 150))
        else:          _txt(frame, "= DUNG YEN", px+10, 78, 0.36, _C["sub"])

        bx, by, bw, bh = px+10, 90, 153, 16
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (40, 40, 60), -1)
        fw = int(bw * ratio)
        if fw > 0: cv2.rectangle(frame, (bx, by), (bx+fw, by+bh), bc, -1)
        mx = bx + int(bw * 0.5)
        cv2.line(frame, (mx, by-2), (mx, by+bh+2), _C["accent"], 2)

        if len(self.vel_graph) > 2:
            gh, gy = 28, 112
            vmax = threshold * 2
            pts = []
            for i, v in enumerate(list(self.vel_graph)[-153:]):
                gx_ = px + 10 + i
                gy_ = gy + int(gh/2 - (v / (vmax+1e-6)) * gh)
                gy_ = max(gy, min(gy+gh, gy_))
                pts.append((gx_, gy_))
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (100, 180, 255), 1)

    def _angle_panel(self, frame, result: DetectionResult):
        if not result.metrics:
            return
        px = self.w - 185
        _rrect(frame, px, 160, px+173, 220, 10, _C["panel"])
        _txt(frame, "GOC CO THE", px+10, 177, 0.36, _C["sub"])
        a  = result.metrics.body_angle
        ac = (255, 160, 30) if a > 65 else ((80, 180, 255) if a > 40 else (80, 220, 80))
        _txt(frame, f"{a:.1f} deg", px+10, 205, 0.75, ac, 2)
        _txt(frame, f"H/W:{result.metrics.aspect_ratio:.2f}  {result.metrics.source}",
             px+10, 218, 0.32, _C["sub"])

    def _fall_alert(self, frame, result: DetectionResult):
        if not result.is_falling:
            self.alert_t = max(0, self.alert_t - 4)
            return
        self.alert_t = min(self.alert_t + 6, 80)
        alpha = (math.sin(self.alert_t * 0.15) * 0.5 + 0.5) * 0.35
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (self.w, self.h), (0, 0, 200), -1)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
        bw_ = int(alpha * 40) + 4
        cv2.rectangle(frame, (0, 0), (self.w, self.h), (0, 0, 200), bw_)

        bx, by = self.w//2 - 200, self.h//2 - 55
        _rrect(frame, bx, by, bx+400, by+110, 14, (15, 8, 8), alpha=0.93)
        cv2.rectangle(frame, (bx, by), (bx+400, by+110), (0, 0, 180), 2)
        blink = (self.alert_t // 10) % 2 == 0
        _txt_c(frame, "! PHAT HIEN TE NGA !", self.w//2, by+32, 0.9,
               (80, 80, 255) if blink else (255, 100, 100), 2)
        _txt_c(frame, "FALL DETECTED", self.w//2, by+55, 0.52, (180, 180, 255))
        _txt_c(frame, f"Toc do: {result.fall_max_velocity:.0f} px/s",
               self.w//2, by+78, 0.5, _C["accent"])
        _txt_c(frame, time.strftime("Luc: %H:%M:%S"), self.w//2, by+98, 0.38, (160, 160, 180))

    def _stats_bar(self, frame, result: DetectionResult, fps: float, backend_ok: bool):
        bh = 34; y0 = self.h - bh
        ov = frame.copy()
        cv2.rectangle(ov, (0, y0), (self.w, self.h), (12, 12, 22), -1)
        cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
        _txt(frame, f"FPS: {fps:.0f}", 10, y0+22, 0.5, _C["sub"])
        fc = (80, 80, 255) if result.fall_count > 0 else _C["sub"]
        _txt(frame, f"Te nga: {result.fall_count} lan", 110, y0+22, 0.5, fc,
             2 if result.fall_count > 0 else 1)
        sv = (80, 220, 80) if backend_ok else (255, 100, 100)
        _txt(frame, "BACKEND: OK" if backend_ok else "BACKEND: OFFLINE",
             self.w - 200, y0+22, 0.38, sv)

    def _face_boxes(self, frame, recognized_persons: List):
        """Vẽ bounding box + tên cho từng khuôn mặt phát hiện được."""
        for p in recognized_persons:
            box   = p.box
            color = _FACE_KNOWN_COLOR if p.is_known else _FACE_UNKNOWN_COLOR
            cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)

            label = f"{p.name}" if not p.is_known else f"{p.name} {p.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(box.y1 - 6, th + 4)
            cv2.rectangle(frame,
                          (box.x1, ly - th - 4),
                          (box.x1 + tw + 6, ly + 2),
                          color, -1)
            cv2.putText(frame, label,
                        (box.x1 + 3, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)

    def render(
        self,
        frame:              np.ndarray,
        result:             DetectionResult,
        fps:                float,
        backend_ok:         bool  = False,
        fall_vel_threshold: float = 80.0,
        recognized_persons: List  = [],
    ) -> np.ndarray:
        self._state_badge(frame, result)
        self._velocity_panel(frame, result, fall_vel_threshold)
        self._angle_panel(frame, result)
        self._face_boxes(frame, recognized_persons)
        self._stats_bar(frame, result, fps, backend_ok)
        self._fall_alert(frame, result)
        return frame
