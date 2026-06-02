"""
src/ui/theme.py — Màu sắc, widget factories dùng chung cho toàn bộ UI.
"""
from __future__ import annotations
import tkinter as tk
from schemas import PoseState

T = {
    "bg"    : "#0c0c1a", "panel" : "#151525", "card"  : "#1c1c30",
    "border": "#28284a", "accent": "#4a9eff", "danger": "#ff3b3b",
    "ok"    : "#3bff8a", "warn"  : "#ffaa22", "text"  : "#e0e0f0",
    "sub"   : "#6868a0", "stand" : "#3bff8a", "sit"   : "#4a9eff",
    "lie"   : "#ffaa22", "walk"  : "#b464ff", "fall"  : "#ff3b3b",
    "face"  : "#50f0c0",
}

STATE_COLOR_TK = {
    PoseState.STANDING: T["stand"], PoseState.SITTING: T["sit"],
    PoseState.LYING   : T["lie"],   PoseState.WALKING: T["walk"],
    PoseState.FALLING : T["fall"],  PoseState.UNKNOWN: T["sub"],
}


def make_card(parent, title="", pady=8):
    f = tk.Frame(parent, bg=T["card"],
                 highlightbackground=T["border"], highlightthickness=1)
    f.pack(fill="x", pady=(0, pady))
    if title:
        tk.Label(f, text=title.upper(), font=("Courier New", 8, "bold"),
                 fg=T["sub"], bg=T["card"], anchor="w").pack(fill="x", padx=12, pady=(8, 2))
    inner = tk.Frame(f, bg=T["card"])
    inner.pack(fill="x", padx=12, pady=(0, 10))
    return inner


def make_btn(parent, text, cmd, bg=None, fg="#000", **kw):
    return tk.Button(parent, text=text, command=cmd,
                     font=("Courier New", 10, "bold"), fg=fg,
                     bg=bg or T["border"], activebackground=T["panel"],
                     relief="flat", cursor="hand2", padx=8, pady=6, **kw)


def face_center_in_bbox(face_box, person_bbox: tuple) -> bool:
    """Kiểm tra tâm khuôn mặt có nằm trong bounding box người không."""
    cx = (face_box.x1 + face_box.x2) // 2
    cy = (face_box.y1 + face_box.y2) // 2
    x1, y1, x2, y2 = person_bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2
