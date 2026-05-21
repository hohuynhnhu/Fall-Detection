"""
src/app.py — Fall Detection Desktop App
Entry point: python app.py
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import queue
import time
import threading
from PIL import Image, ImageTk

from core.camera_worker import CameraWorker, WorkerFrame
from core.overlay import OverlayRenderer, STATE_LABELS_VN, STATE_SKELETON_COLORS
from services.backend_client import BackendClient
from schemas import (
    ThresholdConfig, FeatureConfig, FallEvent, PoseEvent,
    PoseState, EventType, BodyMetricsPayload,
    PersonDetectedPayload, AddFamilyMemberPayload,
)

# ─── Theme ─────────────────────────────────────────────────────────────────────

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


def _beep_fall():
    """Phát 3 tiếng beep cảnh báo trong thread riêng (không block UI)."""
    def _play():
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1000, 200)
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()


def _card(parent, title="", pady=8):
    f = tk.Frame(parent, bg=T["card"],
                 highlightbackground=T["border"], highlightthickness=1)
    f.pack(fill="x", pady=(0, pady))
    if title:
        tk.Label(f, text=title.upper(), font=("Courier New", 8, "bold"),
                 fg=T["sub"], bg=T["card"], anchor="w").pack(fill="x", padx=12, pady=(8, 2))
    inner = tk.Frame(f, bg=T["card"])
    inner.pack(fill="x", padx=12, pady=(0, 10))
    return inner

def _btn(parent, text, cmd, bg=None, fg="#000", **kw):
    return tk.Button(parent, text=text, command=cmd,
                     font=("Courier New", 10, "bold"), fg=fg,
                     bg=bg or T["border"], activebackground=T["panel"],
                     relief="flat", cursor="hand2", padx=8, pady=6, **kw)


# ─── Family Management Window ───────────────────────────────────────────────────

class FamilyManagementWindow(tk.Toplevel):
    """Cửa sổ quản lý thành viên gia đình (thêm / xóa)."""

    def __init__(self, parent_app: "FallDetectionApp"):
        super().__init__(parent_app.root)
        self._app = parent_app
        self.title("Quản lý thành viên gia đình")
        self.configure(bg=T["bg"])
        self.geometry("480x520")
        self.resizable(False, False)
        self._build()
        self._refresh()

    def _build(self):
        tk.Label(self, text="THÀNH VIÊN GIA ĐÌNH",
                 font=("Courier New", 13, "bold"), fg=T["accent"], bg=T["bg"]
                 ).pack(pady=(14, 6))

        # List frame
        lf = tk.Frame(self, bg=T["card"],
                      highlightbackground=T["border"], highlightthickness=1)
        lf.pack(fill="both", expand=True, padx=16, pady=6)

        cols = ("Tên", "Vai trò", "Mẫu", "ID")
        self._tree = ttk.Treeview(lf, columns=cols, show="headings", height=14)
        for c in cols:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=100 if c != "ID" else 80, anchor="center")
        self._tree.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(lf, command=self._tree.yview)
        sb.pack(side="right", fill="y")
        self._tree.configure(yscrollcommand=sb.set)

        # Buttons
        bf = tk.Frame(self, bg=T["bg"])
        bf.pack(fill="x", padx=16, pady=10)
        _btn(bf, "＋ Thêm thành viên",  self._add,    bg=T["ok"],    fg="#000").pack(side="left", padx=4)
        _btn(bf, "✕ Xóa đã chọn",       self._delete, bg=T["danger"], fg="#fff").pack(side="left", padx=4)
        _btn(bf, "↺ Làm mới",            self._refresh, fg=T["text"]).pack(side="right", padx=4)

        # Status
        self._status = tk.Label(self, text="", font=("Courier New", 8),
                                fg=T["sub"], bg=T["bg"])
        self._status.pack(pady=(0, 8))

    def _refresh(self):
        for row in self._tree.get_children():
            self._tree.delete(row)
        members = []
        if self._app._worker and self._app._worker._face_engine:
            members = self._app._worker._face_engine.list_members()
        for m in members:
            self._tree.insert("", "end",
                              values=(m["name"], m["role"], m["sample_count"], m["person_id"]))
        self._status.config(text=f"{len(members)} thành viên trong database")

    def _add(self):
        frame = self._app._last_frame_bgr
        if frame is None:
            messagebox.showwarning("Chưa có camera",
                                   "Bấm BẮT ĐẦU trước, sau đó thêm thành viên.", parent=self)
            return

        fe = self._app._worker._face_engine if self._app._worker else None
        if fe is None:
            messagebox.showwarning("Nhận diện khuôn mặt chưa bật",
                                   "Bật checkbox 'Nhận diện khuôn mặt' trước.", parent=self)
            return

        name = simpledialog.askstring("Tên thành viên", "Nhập tên:", parent=self)
        if not name:
            return
        role = simpledialog.askstring("Vai trò", "Vai trò (family / caregiver):",
                                      initialvalue="family", parent=self) or "family"

        pid = fe.enroll(frame, name=name.strip(), role=role.strip())
        if pid is None:
            messagebox.showerror("Không phát hiện khuôn mặt",
                                 "Không thấy khuôn mặt trong frame hiện tại.\n"
                                 "Hãy đứng trước camera rồi thử lại.", parent=self)
            return

        # Sync metadata lên backend (không bắt buộc — bỏ qua nếu offline)
        payload = AddFamilyMemberPayload(person_id=pid, name=name.strip(), role=role.strip())
        ok = self._app._backend.add_family_member_sync(payload)
        suffix = "" if ok else " (backend offline — lưu local)"
        messagebox.showinfo("Thành công",
                            f"Đã thêm '{name}' (ID: {pid}){suffix}", parent=self)
        self._refresh()

    def _delete(self):
        sel = self._tree.selection()
        if not sel:
            return
        values = self._tree.item(sel[0], "values")
        pid, name = values[3], values[0]
        if not messagebox.askyesno("Xác nhận", f"Xóa '{name}' khỏi database?", parent=self):
            return

        fe = self._app._worker._face_engine if self._app._worker else None
        if fe:
            fe.remove_member(pid)
        self._app._backend.remove_family_member_sync(pid)
        self._refresh()


# ─── App ───────────────────────────────────────────────────────────────────────

class FallDetectionApp:
    def __init__(self, root: tk.Tk):
        self.root   = root
        self.root.title("Fall Detection System")
        self.root.configure(bg=T["bg"])
        self.root.geometry("1440x860")
        self.root.minsize(1100, 680)

        self._running       = False
        self._worker: CameraWorker | None      = None
        self._renderer: OverlayRenderer | None = None
        self._queue: queue.Queue               = queue.Queue(maxsize=3)
        self._config        = ThresholdConfig()
        self._features      = FeatureConfig()
        self._alert_blink        = False
        self._frame_id           = 0
        self._thresh_vars: dict  = {}
        self._last_frame_bgr     = None  # dùng cho enrollment
        self._trans_fall_logged  = False  # debounce AI fall log
        self._last_beep_t        = 0.0   # cooldown beep 3s
        self._lying_start_t      = 0.0   # track lying start for AI confirm logic
        # Debounce gửi person-detected: {person_id: last_sent_time}
        self._person_sent_at: dict[str, float] = {}

        self._backend = BackendClient(
            base_url="http://localhost:8000",
            camera_id="cam_0",
            on_config_update=self._on_backend_config,
            on_feature_update=self._on_backend_features,
        )

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll()

    # ── Build UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header bar ────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg="#080816", height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Frame(hdr, bg=T["accent"], width=4).pack(side="left", fill="y")
        tk.Label(hdr, text="  FALL DETECTION SYSTEM",
                 font=("Courier New", 15, "bold"), fg=T["accent"], bg="#080816"
                 ).pack(side="left", padx=(6, 0), pady=10)
        tk.Label(hdr, text="MediaPipe · YOLO · Transformer AI · Face ID · YAMNet Audio",
                 font=("Courier New", 8), fg=T["sub"], bg="#080816"
                 ).pack(side="left", padx=14)
        self._dot = tk.Label(hdr, text="● OFFLINE",
                              font=("Courier New", 9, "bold"), fg=T["danger"], bg="#080816")
        self._dot.pack(side="right", padx=16)

        # ── Body ──────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=T["bg"])
        body.pack(fill="both", expand=True, padx=12, pady=(8, 12))

        # Right panel (fixed width, scrollable)
        panel_outer = tk.Frame(body, bg=T["bg"], width=330)
        panel_outer.pack(side="right", fill="y", padx=(10, 0))
        panel_outer.pack_propagate(False)

        canvas = tk.Canvas(panel_outer, bg=T["bg"], highlightthickness=0)
        vsb    = tk.Scrollbar(panel_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=T["bg"])
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Camera feed
        cam_border = tk.Frame(body, bg=T["border"])
        cam_border.pack(side="left", fill="both", expand=True)
        self._cam = tk.Label(cam_border, bg="#050510",
                              text="Camera chưa khởi động\nBấm  ▶ BẮT ĐẦU",
                              fg=T["sub"], font=("Courier New", 13))
        self._cam.pack(fill="both", expand=True, padx=1, pady=1)

        self._build_panel(inner)

    def _build_panel(self, panel):
        # ── 1. ĐIỀU KHIỂN ────────────────────────────────────────────────
        ctrl = _card(panel, "ĐIỀU KHIỂN", pady=6)

        # 3 buttons side by side
        br = tk.Frame(ctrl, bg=T["card"]); br.pack(fill="x", pady=(0, 8))
        self._btn_start = _btn(br, "▶  BẮT ĐẦU", self._start, bg=T["ok"])
        self._btn_start.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._btn_stop = _btn(br, "■  DỪNG", self._stop, fg=T["text"])
        self._btn_stop.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._btn_stop.config(state="disabled")
        _btn(br, "↺  RESET", self._reset_falls, fg=T["text"]).pack(side="left", fill="x", expand=True)

        # Nguồn camera / RTSP URL
        sr = tk.Frame(ctrl, bg=T["card"]); sr.pack(fill="x", pady=(0, 2))
        tk.Label(sr, text="Nguồn:", fg=T["sub"], bg=T["card"],
                 font=("Courier New", 8), width=7, anchor="w").pack(side="left")
        self._cam_var = tk.StringVar(value="0")
        tk.Entry(sr, textvariable=self._cam_var,
                 bg=T["panel"], fg=T["text"], insertbackground=T["text"],
                 relief="flat", font=("Courier New", 8)
                 ).pack(side="left", fill="x", expand=True)
        tk.Label(ctrl, text="   0 · 1 · rtsp://ip:8080/h264_ulaw.sdp",
                 fg=T["sub"], bg=T["card"], font=("Courier New", 6), anchor="w"
                 ).pack(fill="x")

        # Checkboxes
        cr = tk.Frame(ctrl, bg=T["card"]); cr.pack(fill="x", pady=(6, 4))
        self._yolo_var = tk.BooleanVar(value=False)
        tk.Checkbutton(cr, text="YOLO", variable=self._yolo_var,
                       bg=T["card"], fg=T["sub"], selectcolor=T["panel"],
                       font=("Courier New", 9)).pack(side="left")
        self._face_var = tk.BooleanVar(value=False)
        tk.Checkbutton(cr, text="Face ID", variable=self._face_var,
                       bg=T["card"], fg=T["face"], selectcolor=T["panel"],
                       font=("Courier New", 9)).pack(side="left", padx=8)
        self._transformer_var = tk.BooleanVar(value=True)
        tk.Checkbutton(cr, text="AI", variable=self._transformer_var,
                       bg=T["card"], fg="#ff9f40", selectcolor=T["panel"],
                       font=("Courier New", 9)).pack(side="left")

        # Backend API URL
        ur = tk.Frame(ctrl, bg=T["card"]); ur.pack(fill="x")
        tk.Label(ur, text="API:", fg=T["sub"], bg=T["card"],
                 font=("Courier New", 8), width=7, anchor="w").pack(side="left")
        self._url_var = tk.StringVar(value="http://localhost:8000")
        tk.Entry(ur, textvariable=self._url_var,
                 bg=T["panel"], fg=T["text"], insertbackground=T["text"],
                 relief="flat", font=("Courier New", 8)
                 ).pack(side="left", fill="x", expand=True)

        # ── 2. TRẠNG THÁI + AI ───────────────────────────────────────────
        sc = _card(panel, "TRẠNG THÁI", pady=6)
        st_row = tk.Frame(sc, bg=T["card"]); st_row.pack(fill="x")

        # Left: rule-based state
        left_col = tk.Frame(st_row, bg=T["card"])
        left_col.pack(side="left", fill="both", expand=True)
        self._state_lbl = tk.Label(left_col, text="---",
            font=("Courier New", 27, "bold"), fg=T["sub"], bg=T["card"], anchor="w")
        self._state_lbl.pack(fill="x")
        self._conf_lbl = tk.Label(left_col, text="Confidence: —",
            font=("Courier New", 8), fg=T["sub"], bg=T["card"], anchor="w")
        self._conf_lbl.pack(fill="x")

        # Divider
        tk.Frame(st_row, bg=T["border"], width=1).pack(side="left", fill="y", padx=8)

        # Right: transformer AI
        right_col = tk.Frame(st_row, bg=T["card"])
        right_col.pack(side="right", anchor="n")
        tk.Label(right_col, text="AI Transformer", font=("Courier New", 7, "bold"),
                 fg="#ff9f40", bg=T["card"]).pack(anchor="e")
        self._ai_lbl = tk.Label(right_col, text="—",
            font=("Courier New", 11, "bold"), fg=T["sub"], bg=T["card"])
        self._ai_lbl.pack(anchor="e")
        self._ai_buf = tk.Label(right_col, text="",
            font=("Courier New", 7), fg=T["sub"], bg=T["card"])
        self._ai_buf.pack(anchor="e")

        # ── 3. CẢNH BÁO TÉ NGÃ ──────────────────────────────────────────
        ac = _card(panel, "CẢNH BÁO TÉ NGÃ", pady=6)
        self._alert_lbl = tk.Label(ac, text="CHƯA PHÁT HIỆN",
            font=("Courier New", 14, "bold"), fg=T["ok"], bg=T["card"])
        self._alert_lbl.pack(pady=(2, 0))
        self._fall_src_lbl = tk.Label(ac, text="",
            font=("Courier New", 7), fg=T["sub"], bg=T["card"])
        self._fall_src_lbl.pack()
        self._fall_cnt = tk.Label(ac, text="Số lần té: 0",
            font=("Courier New", 9), fg=T["text"], bg=T["card"])
        self._fall_cnt.pack(pady=(2, 0))

        # ── 4. ÂM THANH (YAMNet) ────────────────────────────────────────────
        au = _card(panel, "ÂM THANH (YAMNet)", pady=6)
        # Row 1: status + API badge
        au_row1 = tk.Frame(au, bg=T["card"]); au_row1.pack(fill="x")
        self._audio_status_lbl = tk.Label(au_row1, text="KHÔNG KHẢ DỤNG",
            font=("Courier New", 11, "bold"), fg=T["sub"], bg=T["card"], anchor="w")
        self._audio_status_lbl.pack(side="left")
        self._audio_api_badge = tk.Label(au_row1, text="[API: —]",
            font=("Courier New", 7), fg=T["sub"], bg=T["card"], anchor="e")
        self._audio_api_badge.pack(side="right")
        # Row 2: detected sound class + confidence
        self._audio_class_lbl = tk.Label(au, text="",
            font=("Courier New", 8), fg=T["sub"], bg=T["card"], anchor="w")
        self._audio_class_lbl.pack(fill="x")
        self._audio_conf_lbl = tk.Label(au, text="",
            font=("Courier New", 8), fg=T["sub"], bg=T["card"], anchor="w")
        self._audio_conf_lbl.pack(fill="x")

        # ── 5. NHẬN DIỆN KHUÔN MẶT ──────────────────────────────────────
        fc = _card(panel, "NHẬN DIỆN KHUÔN MẶT", pady=6)
        # Face header row with API badge
        fc_row1 = tk.Frame(fc, bg=T["card"]); fc_row1.pack(fill="x")
        self._face_name_lbl = tk.Label(fc_row1, text="Chưa phát hiện",
            font=("Courier New", 12, "bold"), fg=T["sub"], bg=T["card"], anchor="w")
        self._face_name_lbl.pack(side="left")
        self._face_api_badge = tk.Label(fc_row1, text="[API: —]",
            font=("Courier New", 7), fg=T["sub"], bg=T["card"], anchor="e")
        self._face_api_badge.pack(side="right")
        self._face_conf_lbl = tk.Label(fc, text="",
            font=("Courier New", 8), fg=T["sub"], bg=T["card"])
        self._face_conf_lbl.pack()
        _btn(fc, "👥 Quản lý thành viên", self._open_family_mgmt,
             fg=T["text"], bg=T["border"]).pack(fill="x", pady=(6, 0))

        # ── 6. CHỈ SỐ (lưới 2 cột) ──────────────────────────────────────
        mc = _card(panel, "CHỈ SỐ", pady=6)
        metric_grid = [
            ("Vel Y",  "_lbl_vy",    "Vel X",  "_lbl_vx"),
            ("Góc",    "_lbl_ang",   "H/W",    "_lbl_ratio"),
            ("Nguồn",  "_lbl_src",   "FPS",    "_lbl_fps"),
        ]
        for row_data in metric_grid:
            r = tk.Frame(mc, bg=T["card"]); r.pack(fill="x", pady=2)
            for i in [0, 2]:
                tk.Label(r, text=row_data[i]+":", font=("Courier New", 8),
                         fg=T["sub"], bg=T["card"], width=7, anchor="w").pack(side="left")
                lbl = tk.Label(r, text="—", font=("Courier New", 9, "bold"),
                               fg=T["text"], bg=T["card"], width=6, anchor="w")
                lbl.pack(side="left")
                setattr(self, row_data[i + 1], lbl)
                if i == 0:
                    tk.Label(r, text="│", font=("Courier New", 9),
                             fg=T["border"], bg=T["card"]).pack(side="left", padx=4)

        # ── 7. NGƯỠNG PHÁT HIỆN ─────────────────────────────────────────
        tc = _card(panel, "NGƯỠNG PHÁT HIỆN", pady=6)
        for fname, label in [
            ("fall_velocity_threshold", "Fall vel"),
            ("body_angle_lying",        "Góc nằm"),
            ("aspect_ratio_lying",      "H/W nằm"),
            ("fall_confirm_frames",     "Confirm"),
            ("walk_velocity_threshold", "Walk vel"),
        ]:
            r = tk.Frame(tc, bg=T["card"]); r.pack(fill="x", pady=2)
            tk.Label(r, text=label + ":", font=("Courier New", 8), fg=T["sub"],
                     bg=T["card"], width=11, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(getattr(self._config, fname)))
            tk.Entry(r, textvariable=var, width=7,
                     bg=T["panel"], fg=T["text"], insertbackground=T["text"],
                     relief="flat", font=("Courier New", 9)).pack(side="left", padx=4)
            self._thresh_vars[fname] = var
        _btn(tc, "↑  APPLY", self._apply_thresh,
             fg=T["text"], bg=T["border"]).pack(fill="x", pady=(6, 0))

        # ── 8. SỰ KIỆN ──────────────────────────────────────────────────
        lc = _card(panel, "SỰ KIỆN", pady=0)
        log_frame = tk.Frame(lc, bg=T["panel"])
        log_frame.pack(fill="both", expand=True)
        self._log = tk.Text(log_frame, height=8, bg=T["panel"], fg=T["text"],
                             font=("Courier New", 7), relief="flat",
                             state="disabled", wrap="word")
        self._log.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(log_frame, command=self._log.yview)
        sb.pack(side="right", fill="y")
        self._log.configure(yscrollcommand=sb.set)

    # ── Controls ───────────────────────────────────────────────────────────────

    def _start(self):
        if self._running:
            return
        self._backend.base_url = self._url_var.get().rstrip("/")

        cfg = self._backend.fetch_config_sync()
        if cfg:
            self._config   = cfg
            self._features = self._backend.current_features
            self._sync_thresh_ui()
            self._sync_feature_badges()
            self._log_ev("✓ Config + features sync từ backend")
        else:
            self._log_ev("⚠ Backend offline — dùng config mặc định")

        raw_src    = self._cam_var.get().strip()
        cam_source = int(raw_src) if raw_src.isdigit() else raw_src

        self._queue  = queue.Queue(maxsize=3)
        self._worker = CameraWorker(
            camera_source   = cam_source,
            result_queue    = self._queue,
            use_yolo        = self._yolo_var.get(),
            use_face        = self._face_var.get(),
            use_transformer = self._transformer_var.get(),
            config          = self._config,
            features        = self._features,
            camera_id       = "cam_0",
            backend_client  = self._backend,
        )
        self._worker.start()
        self._running = True
        self._backend.start()

        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        face_txt  = " | Face ID" if self._face_var.get()        else ""
        trans_txt = " | AI"      if self._transformer_var.get() else ""
        yolo_txt  = " | YOLO"   if self._yolo_var.get()        else ""
        audio_txt = " | Audio"  if self._features.enable_sound_detection else ""
        src_short = raw_src if len(raw_src) <= 30 else raw_src[:27] + "…"
        self._log_ev(f"▶ {src_short}{yolo_txt}{face_txt}{trans_txt}{audio_txt}")

    def _stop(self):
        if not self._running:
            return
        self._running = False
        if self._worker:
            self._worker.stop()
        self._backend.stop()
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._cam.config(image="",
                         text="Camera dừng\nBấm BẮT ĐẦU để tiếp tục",
                         fg=T["sub"])
        self._log_ev("■ Camera dừng")

    def _reset_falls(self):
        if self._worker:
            self._worker.reset_falls()
        self._fall_cnt.config(text="Số lần té: 0")
        self._alert_lbl.config(text="CHƯA PHÁT HIỆN", fg=T["ok"])
        self._log_ev("↺ Reset fall history")

    def _apply_thresh(self):
        try:
            for fname, var in self._thresh_vars.items():
                setattr(self._config, fname, float(var.get()))
            if self._worker:
                self._worker.update_config(self._config)
            self._log_ev("↑ Thresholds cập nhật")
        except ValueError as e:
            messagebox.showerror("Lỗi", f"Giá trị không hợp lệ: {e}")

    def _sync_thresh_ui(self):
        for fname, var in self._thresh_vars.items():
            var.set(str(getattr(self._config, fname)))

    def _on_backend_config(self, cfg: ThresholdConfig):
        self._config = cfg
        self.root.after(0, self._sync_thresh_ui)
        if self._worker:
            self._worker.update_config(cfg)
        self.root.after(0, lambda: self._log_ev("↓ Config sync từ backend"))

    def _on_backend_features(self, feat: FeatureConfig):
        self._features = feat
        self.root.after(0, self._sync_feature_badges)
        if self._worker:
            self._worker.update_features(feat)
        parts = []
        if not feat.enable_face_recognition: parts.append("Face OFF")
        if not feat.enable_sound_detection:  parts.append("Audio OFF")
        if feat.sleep_as_fall:               parts.append("sleep=fall")
        note = ", ".join(parts) if parts else "all features ON"
        self.root.after(0, lambda: self._log_ev(f"↓ Features từ backend: {note}"))

    def _sync_feature_badges(self):
        feat = self._features
        face_state  = "BẬT" if feat.enable_face_recognition else "TẮT"
        audio_state = "BẬT" if feat.enable_sound_detection  else "TẮT"
        face_color  = T["ok"]   if feat.enable_face_recognition else T["danger"]
        audio_color = T["ok"]   if feat.enable_sound_detection  else T["danger"]
        self._face_api_badge.config(
            text=f"[API: {face_state}]", fg=face_color)
        self._audio_api_badge.config(
            text=f"[API: {audio_state}]", fg=audio_color)
        # Show sleep=fall indicator in audio panel
        if feat.sleep_as_fall:
            self._audio_class_lbl.config(
                text="Chế độ: nằm = té ngã", fg=T["warn"])

    def _open_family_mgmt(self):
        FamilyManagementWindow(self)

    # ── Poll queue ─────────────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                item = self._queue.get_nowait()
                if isinstance(item, dict) and "error" in item:
                    messagebox.showerror("Lỗi camera", item["error"])
                    self._stop()
                    break
                if isinstance(item, WorkerFrame):
                    self._update(item)
        except queue.Empty:
            pass
        self.root.after(14, self._poll)

    def _update(self, wf: WorkerFrame):
        self._frame_id += 1
        result = wf.result
        self._last_frame_bgr = wf.frame_bgr

        h, w = wf.frame_bgr.shape[:2]
        if self._renderer is None or self._renderer.w != w:
            self._renderer = OverlayRenderer(w, h)

        rendered = self._renderer.render(
            wf.frame_bgr.copy(), result, wf.fps,
            backend_ok=self._backend.connected,
            fall_vel_threshold=self._config.fall_velocity_threshold,
            recognized_persons=wf.recognized_persons,
        )

        rgb   = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb)
        cw    = self._cam.winfo_width()  or 960
        ch    = self._cam.winfo_height() or 560
        img.thumbnail((cw, ch), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._cam.config(image=photo, text="")
        self._cam.image = photo

        # State label
        s  = str(result.state)
        sc = STATE_COLOR_TK.get(s, T["sub"])
        self._state_lbl.config(text=STATE_LABELS_VN.get(s, "?"), fg=sc)
        conf = result.metrics.confidence if result.metrics else 0.0
        self._conf_lbl.config(text=f"Confidence: {conf:.0%}")

        # Metrics
        vy = result.velocity_y
        self._lbl_vy.config(
            text=f"{abs(vy):.0f}",
            fg=T["danger"] if abs(vy) > self._config.fall_velocity_threshold else T["text"])
        self._lbl_vx.config(text=f"{abs(result.velocity_x):.0f}")
        if result.metrics:
            self._lbl_ang.config(text=f"{result.metrics.body_angle:.1f}°")
            self._lbl_ratio.config(text=f"{result.metrics.aspect_ratio:.2f}")
            self._lbl_src.config(text=result.metrics.source)
        self._lbl_fps.config(text=f"{wf.fps:.0f}")

        # Backend dot
        ok = self._backend.connected
        self._dot.config(
            text="● ONLINE" if ok else "● OFFLINE",
            fg=T["ok"] if ok else T["danger"])

        # ── Transformer AI display ─────────────────────────────────────────
        tr = wf.transformer_result
        trans_fall = False
        if tr is not None and tr.ready:
            trans_fall = tr.is_fall
            if trans_fall:
                ai_color = T["danger"]
                ai_text  = f"FALL  {tr.confidence:.0%}"
            else:
                ai_color = T["ok"]
                ai_text  = f"SAFE  {tr.confidence:.0%}"
            self._ai_lbl.config(text=ai_text, fg=ai_color)
            if hasattr(self, "_worker") and self._worker and self._worker._transformer:
                buf  = self._worker._transformer.buffer_size
                total = self._worker._transformer.num_frames
                self._ai_buf.config(text=f"Buffer {buf}/{total}")
        elif tr is not None and not tr.ready:
            if hasattr(self, "_worker") and self._worker and self._worker._transformer:
                buf  = self._worker._transformer.buffer_size
                total = self._worker._transformer.num_frames
                self._ai_lbl.config(text=f"Đang nạp ({buf}/{total})", fg=T["sub"])
                self._ai_buf.config(text="")
        else:
            self._ai_lbl.config(text="Tắt", fg=T["sub"])
            self._ai_buf.config(text="")

        # ── Fall alert (rule-based OR transformer AI) ──────────────────────
        # ── Fall alert: Transformer xác nhận trước, rule-based xác nhận sau ──
        rule_fall = result.is_falling
        vel_y_abs = abs(result.velocity_y)

        # Thêm biến theo dõi thời điểm bắt đầu nằm
        current_lying = (str(result.state) == str(PoseState.LYING))
        was_upright   = str(result.prev_state) in (
            str(PoseState.STANDING),
            str(PoseState.WALKING),
            str(PoseState.SITTING),
        )

        # Cập nhật thời điểm bắt đầu nằm
        if current_lying and not (str(result.prev_state) == str(PoseState.LYING)):
            self._lying_start_t = time.time()
        lying_duration = time.time() - self._lying_start_t if current_lying else 0.0

        # Bước 1: Transformer phát hiện FALL không?
        ai_sees_fall = (
            tr is not None and
            tr.ready and
            tr.is_fall and
            vel_y_abs > 20.0 and        # velocity đủ lớn
            current_lying and           # đang nằm
            was_upright and             # trước đó đứng/đi/ngồi
            lying_duration < 2.0        # trong 2s đầu nằm xuống
        )

        # Bước 2: Nếu AI thấy FALL → hỏi rule-based xác nhận
        # Nếu AI không thấy → bỏ qua rule-based
        trans_fall = ai_sees_fall and rule_fall

        # Chỉ báo khi CẢ 2 đồng ý
        any_fall = trans_fall

        if any_fall:
            self._alert_blink = not self._alert_blink
            if rule_fall and trans_fall:
                src = "Rules + AI"
            elif trans_fall:
                src = "AI Transformer"
            else:
                src = "Rule-based"
            self._alert_lbl.config(
                text="⚠  TÉ NGÃ !",
                fg=T["danger"] if self._alert_blink else T["warn"])
            self._fall_src_lbl.config(text=f"Nguồn: {src}", fg=T["sub"])

            # Log transformer-only fall once per trigger
            if trans_fall and not rule_fall:
                if not getattr(self, "_trans_fall_logged", False):
                    self._log_ev(
                        f"⚠ AI TÉ NGÃ [{time.strftime('%H:%M:%S')}]  "
                        f"conf={tr.confidence:.0%}"
                    )
                    self._trans_fall_logged = True
            else:
                self._trans_fall_logged = False
        else:
            self._trans_fall_logged = False
            if result.fall_count == 0:
                self._alert_lbl.config(text="CHƯA PHÁT HIỆN", fg=T["ok"])
                self._fall_src_lbl.config(text="", fg=T["sub"])
            else:
                self._alert_lbl.config(text="ĐÃ PHỤC HỒI", fg=T["warn"])
                self._fall_src_lbl.config(text="", fg=T["sub"])

        self._fall_cnt.config(text=f"Số lần té: {result.fall_count}")

        # ── Audio engine status + result ──────────────────────────────────
        self._update_audio_panel(wf)

        # ── Face recognition UI update ─────────────────────────────────────
        persons = wf.recognized_persons
        if persons:
            # Hiển thị người có confidence cao nhất (hoặc người được nhận diện)
            known = [p for p in persons if p.is_known]
            best  = max(known, key=lambda p: p.confidence) if known else persons[0]
            if best.is_known:
                self._face_name_lbl.config(text=best.name, fg=T["face"])
                self._face_conf_lbl.config(
                    text=f"Confidence: {best.confidence:.0%}  |  ID: {best.person_id}",
                    fg=T["sub"])
                self._send_person_detected(best, wf.timestamp)
            else:
                self._face_name_lbl.config(text="Không nhận ra", fg=T["warn"])
                self._face_conf_lbl.config(text=f"{len(persons)} khuôn mặt", fg=T["sub"])
        else:
            self._face_name_lbl.config(text="Chưa phát hiện", fg=T["sub"])
            self._face_conf_lbl.config(text="", fg=T["sub"])

        # ── Beep cảnh báo khi phát hiện té ───────────────────────────────
        should_beep = result.fall_just_triggered or (trans_fall and not rule_fall
                      and not self._trans_fall_logged)
        if should_beep:
            now_t = time.time()
            if now_t - self._last_beep_t >= 3.0:
                _beep_fall()
                self._last_beep_t = now_t

        # ── Fall event (worker gửi kèm clip_url sau khi upload xong) ─────
        if result.fall_just_triggered:
            self._log_ev(
                f"⚠ TÉ NGÃ [{time.strftime('%H:%M:%S')}]  "
                f"v={result.fall_max_velocity:.0f}px/s  "
                f"from={result.prev_state}  📹 ghi clip…"
            )

        # ── Periodic pose event (mỗi 30 frames) ───────────────────────────
        if self._frame_id % 30 == 0 and result.metrics:
            m = result.metrics
            self._backend.send_pose(PoseEvent(
                event_type        = EventType.POSE_CHANGE,
                camera_id         = "cam_0",
                timestamp         = wf.timestamp,
                state             = result.state,
                prev_state        = result.prev_state,
                velocity_px_per_s = result.velocity_y,
                metrics           = BodyMetricsPayload(
                    body_angle   = m.body_angle,
                    aspect_ratio = m.aspect_ratio,
                    center_x     = m.center_x,
                    center_y     = m.center_y,
                    confidence   = m.confidence,
                    hip_y        = m.hip_y,
                    shoulder_y   = m.shoulder_y,
                    ankle_y      = m.ankle_y,
                ),
                frame_id          = self._frame_id,
            ))

    def _update_audio_panel(self, wf: WorkerFrame):
        """Refresh the YAMNet audio panel on every frame."""
        worker = self._worker

        # Engine status (loading / ready / unavailable / disabled)
        if worker is None or worker._audio_engine is None:
            if not self._features.enable_sound_detection:
                self._audio_status_lbl.config(text="TẮT (API)", fg=T["sub"])
            else:
                self._audio_status_lbl.config(text="KHÔNG KHẢ DỤNG", fg=T["sub"])
            return

        eng = worker._audio_engine
        if eng.is_busy:
            self._audio_status_lbl.config(text="ĐANG NGHE…", fg=T["warn"])
            return

        if not eng.loaded:
            self._audio_status_lbl.config(
                text=f"Đang tải… ({eng.status})", fg=T["sub"])
            return

        # Show the latest audio result if one just arrived
        ar = wf.audio_result
        if ar is not None:
            if ar.detected:
                self._audio_status_lbl.config(text="⚡ ĐÃ PHÁT HIỆN", fg=T["danger"])
                self._audio_class_lbl.config(
                    text=f"Âm thanh: {ar.sound_class}", fg=T["warn"])
                self._audio_conf_lbl.config(
                    text=f"Độ tin cậy: {ar.confidence:.0%}", fg=T["text"])
                self._log_ev(
                    f"🔊 Âm thanh: {ar.sound_class} ({ar.confidence:.0%})")
            else:
                self._audio_status_lbl.config(text="SẴN SÀNG", fg=T["ok"])
                self._audio_class_lbl.config(
                    text=f"Không phát hiện ({ar.sound_class})", fg=T["sub"])
                self._audio_conf_lbl.config(text="", fg=T["sub"])
        elif self._features.sleep_as_fall:
            self._audio_status_lbl.config(text="SẴN SÀNG", fg=T["ok"])
            self._audio_class_lbl.config(text="Chế độ: nằm = té ngã", fg=T["warn"])
        else:
            self._audio_status_lbl.config(text="SẴN SÀNG", fg=T["ok"])

    def _send_person_detected(self, person, timestamp: float):
        """Gửi event person-detected với debounce 5 giây / người."""
        now  = time.time()
        last = self._person_sent_at.get(person.person_id, 0.0)
        if now - last < 5.0:
            return
        self._person_sent_at[person.person_id] = now
        self._backend.send_person_detected(PersonDetectedPayload(
            event_type  = EventType.PERSON_DETECTED,
            camera_id   = "cam_0",
            timestamp   = timestamp,
            person_id   = person.person_id,
            person_name = person.name,
            confidence  = person.confidence,
            frame_id    = self._frame_id,
        ))
        self._log_ev(f"👤 Nhận diện: {person.name} ({person.confidence:.0%})")

    def _log_ev(self, msg: str):
        self._log.config(state="normal")
        self._log.insert("end", f"{time.strftime('%H:%M:%S')}  {msg}\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def _on_close(self):
        self._stop()
        self.root.destroy()


# ─── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    FallDetectionApp(root)
    root.mainloop()
