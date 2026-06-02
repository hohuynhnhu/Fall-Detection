"""
src/core/appearance_tracker.py — Appearance-based person tracking.

Decouples face recognition (expensive) from per-frame identification:
  1. Run face recognition once when a person is first seen.
  2. Track subsequent frames via color histogram + shape similarity.
  3. Re-run face recognition only when appearance shifts significantly.
"""
from __future__ import annotations
import cv2
import numpy as np
import threading
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AppearanceSnapshot:
    color_hist_upper: np.ndarray  # float32, flattened HSV histogram — upper third of bbox
    color_hist_lower: np.ndarray  # float32, flattened HSV histogram — lower third of bbox
    shoulder_width:   float = 0.0
    body_height:      float = 0.0
    aspect_ratio:     float = 1.5  # bbox H/W
    frame_captured:   int   = 0


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _compute_iou(a: tuple, b: tuple) -> float:
    x1 = max(a[0], b[0]);  y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]);  y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Appearance distance helpers ───────────────────────────────────────────────

def _bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya distance in [0, 1]: 0 = identical, 1 = no overlap."""
    h1f = h1.astype(np.float32).reshape(-1, 1)
    h2f = h2.astype(np.float32).reshape(-1, 1)
    return float(cv2.compareHist(h1f, h2f, cv2.HISTCMP_BHATTACHARYYA))


def _color_hist_diff(s1: AppearanceSnapshot, s2: AppearanceSnapshot) -> float:
    """Average Bhattacharyya distance across upper + lower body histograms."""
    d_upper = _bhattacharyya(s1.color_hist_upper, s2.color_hist_upper)
    d_lower = _bhattacharyya(s1.color_hist_lower, s2.color_hist_lower)
    return float((d_upper + d_lower) / 2)


def _shape_diff(s1: AppearanceSnapshot, s2: AppearanceSnapshot) -> float:
    """Normalized shape difference [0, 1]."""
    parts: list[float] = []
    if s1.shoulder_width > 0 and s2.shoulder_width > 0:
        parts.append(abs(s1.shoulder_width - s2.shoulder_width) /
                     max(s1.shoulder_width, s2.shoulder_width))
    if s1.body_height > 0 and s2.body_height > 0:
        parts.append(abs(s1.body_height - s2.body_height) /
                     max(s1.body_height, s2.body_height))
    parts.append(abs(s1.aspect_ratio - s2.aspect_ratio) /
                 max(s1.aspect_ratio, s2.aspect_ratio, 0.01))
    return float(np.mean(parts)) if parts else 0.0


# ── TrackerManager ─────────────────────────────────────────────────────────────

class TrackerManager:
    """
    Maintains per-person identity state across frames.

    track_id is the YOLO ByteTrack person ID (int).  IoU matching in match_box
    handles the case where YOLO reassigns an ID after a brief occlusion.
    Thread-safe via threading.Lock.
    """

    def __init__(self) -> None:
        # track_id → {person_id, name, is_patient, snapshot,
        #              last_seen_frame, last_box, stable_frames}
        self._tracks: Dict[int, dict] = {}
        self._lock = threading.Lock()

    # ── Core API ───────────────────────────────────────────────────────────────

    def match_box(self, box: tuple, cur_frame: int) -> Optional[int]:
        """
        Find the existing track whose last known box has IoU >= 0.4 with `box`.
        Updates last_box, last_seen_frame, and stable_frames counter on match.
        Returns track_id or None.
        """
        best_id:  Optional[int] = None
        best_iou: float         = 0.4  # minimum threshold
        with self._lock:
            for tid, t in self._tracks.items():
                lb = t["last_box"]
                if lb is None:
                    continue
                iou = _compute_iou(box, lb)
                if iou > best_iou:
                    best_iou = iou
                    best_id  = tid
            if best_id is not None:
                self._tracks[best_id]["last_box"]        = box
                self._tracks[best_id]["last_seen_frame"] = cur_frame
                self._tracks[best_id]["stable_frames"]  += 1
        return best_id

    def has_track(self, track_id: int) -> bool:
        with self._lock:
            return track_id in self._tracks

    def add_track(
        self,
        track_id:   int,
        person_id:  Optional[str],
        name:       str,
        is_patient: bool,
        snapshot:   Optional[AppearanceSnapshot],
        box:        Optional[tuple] = None,
        cur_frame:  int = 0,
    ) -> None:
        with self._lock:
            self._tracks[track_id] = {
                "person_id":       person_id,
                "name":            name,
                "is_patient":      is_patient,
                "notify_on_fall":  True,
                "snapshot":        snapshot,
                "last_seen_frame": cur_frame,
                "last_box":        box,
                "stable_frames":   0,
            }

    def update_track_identity(
        self,
        track_id:       int,
        person_id:      str,
        name:           str,
        is_patient:     bool,
        notify_on_fall: bool = True,
    ) -> None:
        """Set recognized identity after face recognition succeeds."""
        with self._lock:
            if track_id in self._tracks:
                t = self._tracks[track_id]
                t["person_id"]      = person_id
                t["name"]           = name
                t["is_patient"]     = is_patient
                t["notify_on_fall"] = notify_on_fall

    def update_last_box(self, track_id: int, box: tuple, cur_frame: int) -> None:
        with self._lock:
            if track_id in self._tracks:
                t = self._tracks[track_id]
                t["last_box"]        = box
                t["last_seen_frame"] = cur_frame
                t["stable_frames"]  += 1

    def update_snapshot(
        self,
        track_id:  int,
        snapshot:  AppearanceSnapshot,
        cur_frame: int,
    ) -> None:
        with self._lock:
            if track_id in self._tracks:
                t = self._tracks[track_id]
                t["snapshot"]        = snapshot
                t["last_seen_frame"] = cur_frame

    def remove_stale(self, cur_frame: int, max_gap: int = 60) -> None:
        """Delete tracks not seen for more than max_gap frames (~2 s at 30 fps)."""
        with self._lock:
            stale = [
                tid for tid, t in self._tracks.items()
                if t["last_seen_frame"] > 0
                and cur_frame - t["last_seen_frame"] > max_gap
            ]
            for tid in stale:
                name = self._tracks[tid].get("name") or f"#{tid}"
                del self._tracks[tid]
                print(f"[Tracker] Track {tid} ({name}) removed (stale >{max_gap} frames)")

    def needs_reidentify(self, track_id: int, new_snapshot: AppearanceSnapshot) -> bool:
        """
        True if appearance changed enough to re-run face recognition.
        Only fires once the track is stable (>= 30 frames seen).
        Triggers when: color_diff (Bhattacharyya) > 0.45 OR shape_diff > 0.25.
        """
        return bool(self._reidentify_reason(track_id, new_snapshot))

    def get_track_info(self, track_id: int) -> Optional[dict]:
        with self._lock:
            t = self._tracks.get(track_id)
            return dict(t) if t is not None else None

    def is_identified(self, track_id: int) -> bool:
        """True if face recognition has resolved a person_id for this track."""
        with self._lock:
            t = self._tracks.get(track_id)
            return t is not None and t["person_id"] is not None

    # ── Internal ───────────────────────────────────────────────────────────────

    def _reidentify_reason(
        self, track_id: int, new_snapshot: AppearanceSnapshot
    ) -> str:
        """
        Returns 'color', 'shape', 'color+shape', or '' (no re-ID needed).
        Reads stored snapshot under lock, then computes metrics outside the lock
        (AppearanceSnapshot objects are replaced atomically, never mutated).
        """
        with self._lock:
            t = self._tracks.get(track_id)
            if t is None or t["snapshot"] is None or t["stable_frames"] < 30:
                return ""
            stored = t["snapshot"]

        try:
            color_diff = _color_hist_diff(stored, new_snapshot)
            shape_diff = _shape_diff(stored, new_snapshot)
        except Exception:
            return ""

        color_over = color_diff > 0.45
        shape_over = shape_diff > 0.25
        if color_over and shape_over:
            return "color+shape"
        if color_over:
            return "color"
        if shape_over:
            return "shape"
        return ""
