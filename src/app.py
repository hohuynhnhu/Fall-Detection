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
import tkinter.filedialog as fd
from pathlib import Path
import cv2
import queue
import time
import threading
from PIL import Image, ImageTk

from core.camera_worker import CameraWorker, WorkerFrame
from core.overlay import OverlayRenderer, STATE_LABELS_VN, STATE_SKELETON_COLORS
from core.fall_decision import FallDecisionEngine
from services.backend_client import BackendClient
from utils.beep_manager import BeepManager
from schemas import (
    ThresholdConfig, FeatureConfig, FallEvent, PoseEvent,
    PoseState, EventType, BodyMetricsPayload,
    PersonDetectedPayload, AddFamilyMemberPayload, PatientPoseEvent,
)

from ui.theme import T, STATE_COLOR_TK, make_card, make_btn, face_center_in_bbox
from ui.family_window import FamilyManagementWindow

# ─── App ───────────────────────────────────────────────────────────────────────

class FallDetectionApp:
    def __init__(self, root: tk.Tk):
        self.root   = root
        self.root.title("Fall Detection System")
        self.root.configure(bg=T["bg"])
        self.root.geometry("1440x860")
        self.root.minsize(1100, 680)

        self._running       = False
        self._video_source: "int | str" = 0   # 0 = webcam, str = video file path
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
        self._fall_engine        = FallDecisionEngine()
        self._beep               = BeepManager()
        # Debounce gửi person-detected: {person_id: last_sent_time}
        self._person_sent_at: dict[str, float] = {}

        # Theo dõi bệnh nhân: trạng thái và thời điểm gửi cuối
        self._patient_last_state: dict[str, PoseState] = {}
        self._patient_last_sent:  dict[str, float]     = {}

        self._last_sent_pose_state: PoseState = PoseState.UNKNOWN
        self._last_sent_pose_time: float = 0.0

        self._last_known_patient: tuple = None  # (person_id, name, notify_on_fall)

        self._family_manager = None   # khởi tạo trong _start()

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
        self.root.state("zoomed")   # Phóng to toàn màn hình

        # ── Header (40px) ────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg="#07071a", height=40)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Frame(hdr, bg=T["accent"], width=3).pack(side="left", fill="y")
        tk.Label(hdr, text="  FALL DETECTION SYSTEM",
                 font=("Courier New", 13, "bold"), fg=T["accent"], bg="#07071a"
                 ).pack(side="left", padx=(6, 0))
        tk.Label(hdr, text="  —  MediaPipe · YOLO · Transformer AI · Face ID · YAMNet",
                 font=("Courier New", 8), fg=T["sub"], bg="#07071a"
                 ).pack(side="left")
        self._dot = tk.Label(hdr, text="● OFFLINE",
                             font=("Courier New", 9, "bold"), fg=T["danger"], bg="#07071a")
        self._dot.pack(side="right", padx=14)
        self._time_lbl = tk.Label(hdr, text="",
                                  font=("Courier New", 9), fg=T["sub"], bg="#07071a")
        self._time_lbl.pack(side="right", padx=8)

        # ── Body ──────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg="#000")
        body.pack(fill="both", expand=True)

        # Camera feed (trái, chiếm toàn bộ không gian còn lại)
        self._cam = tk.Label(body, bg="#050510",
                             text="Camera chưa khởi động\nBấm  ▶ BẮT ĐẦU",
                             fg=T["sub"], font=("Courier New", 14))
        self._cam.pack(side="left", fill="both", expand=True)

        # Right panel (cố định 280px, không cuộn)
        panel = tk.Frame(body, bg=T["panel"], width=280)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        self._build_panel(panel)

    def _build_panel(self, panel):
        BG = T["panel"]

        def _sep():
            tk.Frame(panel, bg=T["border"], height=1).pack(fill="x")

        def _section(title):
            f = tk.Frame(panel, bg=BG)
            f.pack(fill="x", padx=14, pady=(10, 6))
            if title:
                tk.Label(f, text=title, font=("Courier New", 7, "bold"),
                         fg=T["sub"], bg=BG, anchor="w").pack(fill="x", pady=(0, 4))
            return f

        # ── 1. ĐIỀU KHIỂN ────────────────────────────────────────────────
        ctrl = _section("ĐIỀU KHIỂN")
        br = tk.Frame(ctrl, bg=BG); br.pack(fill="x", pady=(0, 4))
        self._btn_start = make_btn(br, "▶  BẮT ĐẦU", self._start, bg=T["ok"])
        self._btn_start.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._btn_stop = make_btn(br, "■  DỪNG", self._stop, fg=T["text"])
        self._btn_stop.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._btn_stop.config(state="disabled")
        make_btn(br, "↺", self._reset_falls, fg=T["text"]).pack(side="left")

        src_row = tk.Frame(ctrl, bg=BG); src_row.pack(fill="x", pady=(0, 6))
        make_btn(src_row, " Chọn video", self._pick_video, fg=T["text"]).pack(
            side="left", fill="x", expand=True, padx=(0, 3))
        make_btn(src_row, "📷 Webcam", self._set_webcam, fg=T["text"]).pack(
            side="left", fill="x", expand=True)

        # Badges: YOLO · AI · Face (tự động, chỉ hiển thị)
        badge_row = tk.Frame(ctrl, bg=BG); badge_row.pack(fill="x", pady=(0, 6))
        self._badge_yolo  = tk.Label(badge_row, text="● YOLO",  font=("Courier New", 8, "bold"),
                                     fg=T["sub"], bg=BG)
        self._badge_yolo.pack(side="left", padx=(0, 6))
        self._badge_ai    = tk.Label(badge_row, text="● AI",    font=("Courier New", 8, "bold"),
                                     fg=T["sub"], bg=BG)
        self._badge_ai.pack(side="left", padx=(0, 6))
        self._badge_face  = tk.Label(badge_row, text="● FACE",  font=("Courier New", 8, "bold"),
                                     fg=T["sub"], bg=BG)
        self._badge_face.pack(side="left")

        ur = tk.Frame(ctrl, bg=BG); ur.pack(fill="x")
        tk.Label(ur, text="API:", fg=T["sub"], bg=BG,
                 font=("Courier New", 8), width=5, anchor="w").pack(side="left")
        self._url_var = tk.StringVar(value="http://localhost:8000")
        tk.Entry(ur, textvariable=self._url_var, bg=T["card"], fg=T["text"],
                 insertbackground=T["text"], relief="flat", font=("Courier New", 8), bd=4
                 ).pack(side="left", fill="x", expand=True)

        _sep()

        # ── 2. TRẠNG THÁI + AI ───────────────────────────────────────────
        sc = _section("TRẠNG THÁI")
        row = tk.Frame(sc, bg=BG); row.pack(fill="x")
        self._state_lbl = tk.Label(row, text="---",
            font=("Courier New", 28, "bold"), fg=T["sub"], bg=BG, anchor="w")
        self._state_lbl.pack(side="left")

        ai_col = tk.Frame(row, bg=BG); ai_col.pack(side="right", anchor="s", pady=2)
        tk.Label(ai_col, text="AI", font=("Courier New", 7, "bold"),
                 fg="#ff9f40", bg=BG).pack(anchor="e")
        self._ai_lbl = tk.Label(ai_col, text="—",
            font=("Courier New", 11, "bold"), fg=T["sub"], bg=BG)
        self._ai_lbl.pack(anchor="e")
        self._ai_buf = tk.Label(ai_col, text="",
            font=("Courier New", 7), fg=T["sub"], bg=BG)
        self._ai_buf.pack(anchor="e")

        self._conf_lbl = tk.Label(sc, text="Confidence: —",
            font=("Courier New", 8), fg=T["sub"], bg=BG, anchor="w")
        self._conf_lbl.pack(fill="x")

        _sep()

        # ── 3. CẢNH BÁO TÉ NGÃ ──────────────────────────────────────────
        ac = _section("CẢNH BÁO TÉ NGÃ")
        self._alert_lbl = tk.Label(ac, text="CHƯA PHÁT HIỆN",
            font=("Courier New", 13, "bold"), fg=T["ok"], bg=BG, anchor="w")
        self._alert_lbl.pack(fill="x")
        self._fall_src_lbl = tk.Label(ac, text="",
            font=("Courier New", 7), fg=T["sub"], bg=BG, anchor="w")
        self._fall_src_lbl.pack(fill="x")
        self._fall_cnt = tk.Label(ac, text="Số lần té: 0",
            font=("Courier New", 8), fg=T["text"], bg=BG, anchor="w")
        self._fall_cnt.pack(fill="x")

        _sep()

        # ── 4. NHẬN DIỆN KHUÔN MẶT ──────────────────────────────────────
        fc = _section("NHẬN DIỆN KHUÔN MẶT")
        fc_hdr = tk.Frame(fc, bg=BG); fc_hdr.pack(fill="x")
        self._face_name_lbl = tk.Label(fc_hdr, text="Chưa phát hiện",
            font=("Courier New", 12, "bold"), fg=T["sub"], bg=BG, anchor="w")
        self._face_name_lbl.pack(side="left")
        self._face_api_badge = tk.Label(fc_hdr, text="[API: —]",
            font=("Courier New", 7), fg=T["sub"], bg=BG)
        self._face_api_badge.pack(side="right")
        self._face_conf_lbl = tk.Label(fc, text="",
            font=("Courier New", 8), fg=T["sub"], bg=BG, anchor="w")
        self._face_conf_lbl.pack(fill="x")
        self._face_status_lbl = tk.Label(fc, text="⚙ Chờ khởi động...",
            font=("Courier New", 7), fg=T["sub"], bg=BG, anchor="w")
        self._face_status_lbl.pack(fill="x")

        _sep()

        # ── 5. BỆNH NHÂN THEO DÕI ───────────────────────────────────────
        pc = _section("BỆNH NHÂN THEO DÕI")
        pc_hdr = tk.Frame(pc, bg=BG); pc_hdr.pack(fill="x")
        tk.Label(pc_hdr, text="Thông báo tư thế",
                 font=("Courier New", 8), fg=T["sub"], bg=BG, anchor="w").pack(side="left")
        self._patient_notify_badge = tk.Label(pc_hdr, text="[API: —]",
            font=("Courier New", 7), fg=T["sub"], bg=BG)
        self._patient_notify_badge.pack(side="right")
        self._patient_text = tk.Text(
            pc, height=3, bg=T["card"], fg=T["text"],
            font=("Courier New", 8), relief="flat", state="disabled", wrap="none",
        )
        self._patient_text.pack(fill="x", pady=(4, 0))
        self._patient_text.tag_configure("stand", foreground=T["stand"])
        self._patient_text.tag_configure("sit",   foreground=T["sit"])
        self._patient_text.tag_configure("lie",   foreground=T["lie"])
        self._patient_text.tag_configure("walk",  foreground=T["walk"])
        self._patient_text.tag_configure("fall",  foreground=T["fall"])
        self._patient_text.tag_configure("dim",   foreground=T["sub"])

        _sep()

        # # ── 6. CHỈ SỐ ────────────────────────────────────────────────────
        # mc = _section("CHỈ SỐ")
        # for row_data in [
        #     ("Vel Y", "_lbl_vy",  "Vel X", "_lbl_vx"),
        #     ("Góc",   "_lbl_ang", "H/W",   "_lbl_ratio"),
        #     ("Src",   "_lbl_src", "FPS",   "_lbl_fps"),
        # ]:
        #     r = tk.Frame(mc, bg=BG); r.pack(fill="x", pady=1)
        #     for i in [0, 2]:
        #         tk.Label(r, text=row_data[i]+":", font=("Courier New", 7),
        #                  fg=T["sub"], bg=BG, width=6, anchor="w").pack(side="left")
        #         lbl = tk.Label(r, text="—", font=("Courier New", 8, "bold"),
        #                        fg=T["text"], bg=BG, width=7, anchor="w")
        #         lbl.pack(side="left")
        #         setattr(self, row_data[i + 1], lbl)
        #         if i == 0:
        #             tk.Label(r, text="│", font=("Courier New", 7),
        #                      fg=T["border"], bg=BG).pack(side="left", padx=2)
        #
        # _sep()
        #
        # # ── 7. ÂM THANH (YAMNet) ────────────────────────────────────────
        # au = _section("ÂM THANH (YAMNet)")
        # au_hdr = tk.Frame(au, bg=BG); au_hdr.pack(fill="x")
        # self._audio_status_lbl = tk.Label(au_hdr, text="KHÔNG KHẢ DỤNG",
        #     font=("Courier New", 10, "bold"), fg=T["sub"], bg=BG, anchor="w")
        # self._audio_status_lbl.pack(side="left")
        # self._audio_api_badge = tk.Label(au_hdr, text="[API: —]",
        #     font=("Courier New", 7), fg=T["sub"], bg=BG)
        # self._audio_api_badge.pack(side="right")
        # self._audio_class_lbl = tk.Label(au, text="",
        #     font=("Courier New", 7), fg=T["sub"], bg=BG, anchor="w")
        # self._audio_class_lbl.pack(fill="x")
        # self._audio_conf_lbl = tk.Label(au, text="",
        #     font=("Courier New", 7), fg=T["sub"], bg=BG, anchor="w")
        # self._audio_conf_lbl.pack(fill="x")

        # _sep()

        # ── 8. SỰ KIỆN (expand, chiếm phần còn lại) ─────────────────────
        lc = tk.Frame(panel, bg=BG)
        lc.pack(fill="both", expand=True, padx=14, pady=(10, 4))
        tk.Label(lc, text="SỰ KIỆN", font=("Courier New", 7, "bold"),
                 fg=T["sub"], bg=BG, anchor="w").pack(fill="x")
        log_frame = tk.Frame(lc, bg=T["card"])
        log_frame.pack(fill="both", expand=True, pady=(4, 0))
        self._log = tk.Text(log_frame, bg=T["card"], fg=T["text"],
                            font=("Courier New", 7), relief="flat",
                            state="disabled", wrap="word")
        self._log.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(log_frame, command=self._log.yview)
        sb.pack(side="right", fill="y")
        self._log.configure(yscrollcommand=sb.set)

        # Khởi tạo thresh_vars (dùng nội bộ, không hiển thị)
        for fname in ["fall_velocity_threshold", "body_angle_lying",
                      "aspect_ratio_lying", "fall_confirm_frames",
                      "walk_velocity_threshold"]:
            self._thresh_vars[fname] = tk.StringVar(
                value=str(getattr(self._config, fname)))

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

        # Fix 1: model_complexity=0 → MediaPipe nhanh nhất (~8ms vs ~25ms)
        self._config.model_complexity = 0

        cam_source = self._video_source

        # Bật face recognition tự động khi backend online (không cần thao tác desktop)
        # Mobile đăng ký bệnh nhân → WS push → desktop tự nhận diện
        use_face, self._family_manager = self._init_face_recognition()

        # Fix 2: queue=1 → luôn hiển thị frame MỚI NHẤT, không bị tích lũy độ trễ
        self._queue  = queue.Queue(maxsize=1)
        self._worker = CameraWorker(
            camera_source   = cam_source,
            result_queue    = self._queue,
            use_yolo        = True,    # luôn bật YOLO
            use_face        = use_face,
            use_transformer = True,    # luôn bật AI Transformer
            config          = self._config,
            features        = self._features,
            camera_id       = "cam_0",
            backend_client  = self._backend,
            family_manager  = self._family_manager,
        )
        self._worker.start()
        self._running = True
        self._backend.start()

        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")

        # Cập nhật badges (YOLO · AI · Face)
        self._badge_yolo.config(fg=T["ok"])
        self._badge_ai.config(fg="#ff9f40")
        self._badge_face.config(fg=T["face"] if use_face else T["sub"])

        # Cập nhật face status label
        if use_face:
            self._face_status_lbl.config(
                text="⚙ Tự động · đang tải thành viên...", fg=T["sub"])
        else:
            self._face_status_lbl.config(
                text="○ Không có thành viên đăng ký ảnh", fg=T["sub"])

        audio_txt = " | Audio" if self._features.enable_sound_detection else ""
        face_txt  = " | Face ID ✓" if use_face else " | Face ID (chưa có thành viên)"
        self._log_ev(f"▶ Camera — YOLO | AI{face_txt}{audio_txt}")

    def _init_face_recognition(self):
        """
        Tự động kiểm tra backend:
        - Có thành viên nào đã đăng ký ảnh khuôn mặt → bật Face ID
        - Không có → tắt Face ID, tiết kiệm tài nguyên
        Trả về (use_face: bool, FamilyManager | None).
        """
        import requests as _req
        try:
            r = _req.get(
                f"{self._backend.base_url}/family-members/all", timeout=3)
            if r.status_code == 200:
                data    = r.json()
                members = data if isinstance(data, list) else data.get("members", [])
                has_faces = any(m.get("face_image_url") for m in members)
                if not has_faces:
                    self._log_ev(
                        f"ℹ Face ID: {len(members)} thành viên nhưng chưa có ảnh — tắt")
                    return False, None
                self._log_ev(
                    f"✓ Face ID: phát hiện {sum(1 for m in members if m.get('face_image_url'))}"
                    f"/{len(members)} thành viên có ảnh — bật tự động")
            else:
                self._log_ev(f"⚠ Face ID: backend trả HTTP {r.status_code} — tắt")
                return False, None
        except Exception as e:
            self._log_ev(f"⚠ Face ID: không kết nối được backend ({e}) — tắt")
            return False, None

        from core.family_manager import FamilyManager
        fm = FamilyManager(
            base_url=self._backend.base_url,
            on_log=lambda msg: self.root.after(0, lambda m=msg: self._log_ev(m)),
        )
        return True, fm

    def _stop(self):
        if not self._running:
            return
        self._running = False
        if self._worker:
            self._worker.stop()
        if self._family_manager:
            self._family_manager.stop()
            self._family_manager = None
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
        self._fall_engine.reset()
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
        if not feat.enable_face_recognition:          parts.append("Face OFF")
        if not feat.enable_patient_pose_notification: parts.append("Notify OFF")
        if not feat.enable_sound_detection:           parts.append("Audio OFF")
        if feat.sleep_as_fall:                        parts.append("sleep=fall")
        note = ", ".join(parts) if parts else "all features ON"
        self.root.after(0, lambda: self._log_ev(f"↓ Features từ backend: {note}"))

    def _sync_feature_badges(self):
        feat = self._features
        face_state = "BẬT" if feat.enable_face_recognition else "TẮT"
        notify_state = "BẬT" if feat.enable_patient_pose_notification else "TẮT"
        face_color = T["ok"] if feat.enable_face_recognition else T["danger"]
        notify_color = T["ok"] if feat.enable_patient_pose_notification else T["danger"]
        self._face_api_badge.config(
            text=f"[API: {face_state}]", fg=face_color)
        self._patient_notify_badge.config(
            text=f"[API: {notify_state}]", fg=notify_color)

    def _open_family_mgmt(self):
        FamilyManagementWindow(self)

    def _pick_video(self):
        path = fd.askopenfilename(
            title="Chọn video demo",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")],
        )
        if path:
            self._video_source = path
            self._log_ev(f"📹 Video: {Path(path).name}")
        else:
            self._video_source = 0
            self._log_ev("📷 Nguồn: Webcam")

    def _set_webcam(self):
        self._video_source = 0
        self._log_ev("📷 Nguồn: Webcam")

    # ── Poll queue ─────────────────────────────────────────────────────────────

    def _poll(self):
        # Cập nhật đồng hồ
        import datetime as _dt
        self._time_lbl.config(text=_dt.datetime.now().strftime("%H:%M:%S"))

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
        self._last_frame_bgr = wf.frame_bgr

        self._render_frame(wf)
        self._update_state_panel(wf)
        self._update_ai_panel(wf)
        self._update_fall_panel(wf)
        self._update_face_panel(wf)

        self._update_patient_monitoring(wf)
        self._maybe_send_pose_event(wf)
        self._send_person_detected(wf)

    def _render_frame(self, wf: WorkerFrame):
        ih, iw = wf.frame_bgr.shape[:2]
        cw = self._cam.winfo_width()  or 1200
        ch = self._cam.winfo_height() or 800
        scale = min(cw * 0.88 / iw, ch * 0.88 / ih)
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        base  = cv2.resize(wf.frame_bgr, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if self._renderer is None or self._renderer.w != new_w:
            self._renderer = OverlayRenderer(new_w, new_h)
        rendered = self._renderer.render(
            base, wf.result, wf.fps,
            backend_ok          = self._backend.connected,
            fall_vel_threshold  = self._config.fall_velocity_threshold,
            recognized_persons  = wf.recognized_persons,
        )
        rgb   = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._cam.config(image=photo, text="")
        self._cam.image = photo

    def _update_state_panel(self, wf: WorkerFrame):
        result = wf.result
        s = str(result.state)
        sc = STATE_COLOR_TK.get(s, T["sub"])
        self._state_lbl.config(text=STATE_LABELS_VN.get(s, "?"), fg=sc)
        conf = result.metrics.confidence if result.metrics else 0.0
        self._conf_lbl.config(text=f"Confidence: {conf:.0%}")

        ok = self._backend.connected
        self._dot.config(
            text="● ONLINE" if ok else "● OFFLINE",
            fg=T["ok"] if ok else T["danger"])
    def _update_ai_panel(self, wf: WorkerFrame):
        tr = wf.transformer_result
        if tr is not None and tr.ready:
            if tr.is_fall:
                self._ai_lbl.config(text=f"FALL  {tr.confidence:.0%}", fg=T["danger"])
            else:
                self._ai_lbl.config(text=f"SAFE  {tr.confidence:.0%}", fg=T["ok"])
            if self._worker and self._worker._transformer:
                buf   = self._worker._transformer.buffer_size
                total = self._worker._transformer.num_frames
                self._ai_buf.config(text=f"Buffer {buf}/{total}")
        elif tr is not None and not tr.ready:
            if self._worker and self._worker._transformer:
                buf   = self._worker._transformer.buffer_size
                total = self._worker._transformer.num_frames
                self._ai_lbl.config(text=f"Đang nạp ({buf}/{total})", fg=T["sub"])
            self._ai_buf.config(text="")
        else:
            self._ai_lbl.config(text="Tắt", fg=T["sub"])
            self._ai_buf.config(text="")

    def _update_fall_panel(self, wf: WorkerFrame):
        result = wf.result
        tr = wf.transformer_result
        vel_y_abs = abs(result.velocity_y)

        decision = self._fall_engine.update(result, tr, vel_y_abs)

        if decision.final_fall:
            self._alert_blink = not self._alert_blink
            self._alert_lbl.config(
                text="⚠  TÉ NGÃ !",
                fg=T["danger"] if self._alert_blink else T["warn"])
            self._fall_src_lbl.config(text=f"Nguồn: {decision.source}", fg=T["sub"])

            if decision.ai_sees_fall and not decision.rule_fall:
                if not self._trans_fall_logged:
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

        should_beep = result.fall_just_triggered or (
                decision.final_fall and not decision.rule_fall and not self._trans_fall_logged
        )
        self._beep.trigger(should_beep)

        if result.fall_just_triggered:
            self._log_ev(
                f"⚠ TÉ NGÃ [{time.strftime('%H:%M:%S')}]  "
                f"v={result.fall_max_velocity:.0f}px/s  "
                f"from={result.prev_state}  📹 ghi clip…"
            )

    def _update_face_panel(self, wf: WorkerFrame):
        persons = wf.recognized_persons
        if persons:
            known = [p for p in persons if p.is_known]
            best = max(known, key=lambda p: p.confidence) if known else persons[0]
            if best.is_known:
                self._face_name_lbl.config(text=best.name, fg=T["face"])
                self._face_conf_lbl.config(
                    text=f"Confidence: {best.confidence:.0%}  |  ID: {best.person_id}",
                    fg=T["sub"])
                # Lưu lại patient cuối cùng được nhận ra
                if best.is_patient:
                    self._last_known_patient = (
                        best.person_id, best.name, best.notify_on_fall
                    )
            else:
                self._face_name_lbl.config(text="Không nhận ra", fg=T["warn"])
                self._face_conf_lbl.config(text=f"{len(persons)} khuôn mặt", fg=T["sub"])
        else:
            self._face_name_lbl.config(text="Chưa phát hiện", fg=T["sub"])
            self._face_conf_lbl.config(text="", fg=T["sub"])

    def _maybe_send_pose_event(self, wf: WorkerFrame):
        result = wf.result
        if not result.metrics:
            return

        now = time.time()
        state_changed = result.state != self._last_sent_pose_state
        heartbeat_due = now - self._last_sent_pose_time >= 10.0

        # Chỉ gửi khi state đổi HOẶC heartbeat 10s
        if not state_changed and not heartbeat_due:
            return

        # Bỏ qua UNKNOWN — không có ý nghĩa gửi lên backend
        if result.state == PoseState.UNKNOWN:
            return

        self._last_sent_pose_state = result.state
        self._last_sent_pose_time = now

        m = result.metrics
        self._backend.send_pose(PoseEvent(
            event_type=EventType.POSE_CHANGE,
            camera_id="cam_0",
            timestamp=wf.timestamp,
            state=result.state,
            prev_state=result.prev_state,
            velocity_px_per_s=result.velocity_y,
            metrics=BodyMetricsPayload(
                body_angle=m.body_angle,
                aspect_ratio=m.aspect_ratio,
                center_x=m.center_x,
                center_y=m.center_y,
                confidence=m.confidence,
                hip_y=m.hip_y,
                shoulder_y=m.shoulder_y,
                ankle_y=m.ankle_y,
            ),
            frame_id=self._frame_id,
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

    def _send_person_detected(self, wf: "WorkerFrame"):
        """Gửi event person-detected (debounce 5s) — đếm số người trong frame."""
        now = time.time()
        if now - self._person_sent_at.get("__cam__", 0.0) < 5.0:
            return

        # Đếm số người từ YOLO multi-person hoặc single-person pose
        if wf.all_person_results:
            count = len(wf.all_person_results)
            conf  = max(
                (r.result.metrics.confidence for r in wf.all_person_results
                 if r.result.metrics), default=0.0
            )
        elif wf.result.state != PoseState.UNKNOWN and wf.result.metrics:
            count = 1
            conf  = wf.result.metrics.confidence
        else:
            return  # không thấy ai

        self._person_sent_at["__cam__"] = now
        self._backend.send_person_detected(PersonDetectedPayload(
            camera_id    = "cam_0",
            timestamp    = wf.timestamp,
            confidence   = conf,
            person_count = count,
        ))

    # ── Patient monitoring ────────────────────────────────────────────────────

    def _get_patient_pose(self, person, wf: "WorkerFrame") -> PoseState:
        """Khớp khuôn mặt với kết quả pose gần nhất (ưu tiên YOLO multi-person)."""
        if wf.all_person_results:
            for pr in wf.all_person_results:
                if face_center_in_bbox(person.box, pr.bbox):
                    return pr.result.state
            # Fallback: lấy người gần tâm khuôn mặt nhất
            fcx = (person.box.x1 + person.box.x2) // 2
            fcy = (person.box.y1 + person.box.y2) // 2
            best_dist, best_state = float("inf"), PoseState.UNKNOWN
            for pr in wf.all_person_results:
                x1, y1, x2, y2 = pr.bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                dist = ((fcx - cx) ** 2 + (fcy - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_state = dist, pr.result.state
            if best_dist < 300:
                return best_state
        return wf.result.state  # single-person fallback

    def _update_patient_monitoring(self, wf: "WorkerFrame"):
        now = time.time()
        patient_rows: list = []

        # ── Ưu tiên: dùng face recognition nếu nhận ra được ─────────────
        if wf.recognized_persons:
            for person in wf.recognized_persons:
                if not person.is_known or not person.is_patient:
                    continue
                self._send_patient_pose_for_person(
                    person.person_id, person.name,
                    person.notify_on_fall, wf, now, patient_rows
                )

        # ── Fallback: mặt bị mất (nằm/quay lưng) nhưng YOLO vẫn thấy người
        elif (self._last_known_patient is not None
              and (wf.result.state != PoseState.UNKNOWN
                   or bool(wf.all_person_results))):
            pid, pname, pnotify = self._last_known_patient
            self._send_patient_pose_for_person(
                pid, pname, pnotify, wf, now, patient_rows
            )

        self._update_patient_panel(patient_rows)

    def _send_patient_pose_for_person(self, pid, pname, notify_on_fall,
                                      wf: "WorkerFrame", now: float,
                                      patient_rows: list):
        """Gửi PatientPoseEvent cho 1 bệnh nhân, có debounce."""
        if not self._features.enable_patient_pose_notification:
            return
        if not notify_on_fall:
            return

        current_state = self._get_patient_pose_by_pid(pid, wf)
        last_state = self._patient_last_state.get(pid)
        last_sent = self._patient_last_sent.get(pid, 0.0)

        display_state = current_state if current_state != PoseState.UNKNOWN else (
                last_state or PoseState.UNKNOWN
        )
        patient_rows.append((pname, display_state, now))

        # # FIX 1: người mới detect (last_state=None) → lưu state, không gửi ngay
        # if last_state is None:
        #     self._patient_last_state[pid] = current_state
        #     self._patient_last_sent[pid] = now
        #     print(f"[Patient] NEW pid={pid} state={current_state}")  # DEBUG
        #     return
        #
        # # FIX 2: debounce 2 giây
        # if now - last_sent < 8.0:
        #     return
        #
        # # FIX 3: skip nếu state giống lần trước
        # if current_state == last_state:
        #     return
        #
        #
        #     # FIX 4: skip UNKNOWN, WALKING và SITTING (hiển thị UI nhưng không gửi notification)
        # # FIX 4: skip UNKNOWN và WALKING
        # if current_state in (PoseState.UNKNOWN, PoseState.WALKING):
        #     return

        # FIX 1: người mới detect → gửi luôn state đầu tiên
        if last_state is None:
            self._patient_last_state[pid] = PoseState.UNKNOWN
            self._patient_last_sent[pid] = 0.0
            print(f"[Patient] NEW pid={pid} state={current_state}")
            # không return → xuống FIX 2,3,4 bình thường

        # FIX 2: debounce 8 giây
        if now - last_sent < 8.0:
            return

        # FIX 3: skip nếu state giống lần trước
        if current_state == last_state:
            return

        # FIX 4: skip UNKNOWN và WALKING
        if current_state in (PoseState.UNKNOWN, PoseState.WALKING):
            return


        self._patient_last_state[pid] = current_state
        self._patient_last_sent[pid] = now
        print(f"[Patient] SEND pid={pid} {last_state}→{current_state} gap={now - last_sent:.1f}s")  # DEBUG

        m = wf.result.metrics
        event = PatientPoseEvent(
            event_type=EventType.PATIENT_POSE,
            camera_id="cam_0",
            timestamp=wf.timestamp,
            person_id=pid,
            person_name=pname,
            state=current_state,
            prev_state=last_state if last_state is not None else PoseState.UNKNOWN,
            metrics=BodyMetricsPayload(
                body_angle=m.body_angle if m else 0.0,
                aspect_ratio=m.aspect_ratio if m else 0.0,
                center_x=m.center_x if m else 0.0,
                center_y=m.center_y if m else 0.0,
                confidence=m.confidence if m else 0.0,
                hip_y=m.hip_y if m else 0.0,
                shoulder_y=m.shoulder_y if m else 0.0,
                ankle_y=m.ankle_y if m else 0.0,
            ),
            frame_id=self._frame_id,
        )
        self._backend.send_patient_pose(event)

        lbl_old = STATE_LABELS_VN.get(last_state, "?")
        lbl_new = STATE_LABELS_VN.get(current_state, "?")
        self._log_ev(f"🏥 {pname}: {lbl_old} → {lbl_new}")

    def _get_patient_pose_by_pid(self, pid, wf: "WorkerFrame") -> PoseState:
        """Lấy pose state của bệnh nhân theo person_id, fallback về wf.result.state."""
        if wf.all_person_results:
            # Tìm theo person_id nếu có face match
            for person in wf.recognized_persons:
                if person.person_id == pid:
                    return self._get_patient_pose(person, wf)
            # Không tìm thấy face → dùng result tổng
            return wf.result.state
        return wf.result.state

    _STATE_TAG = {
        PoseState.STANDING: "stand",
        PoseState.SITTING: "sit",
        PoseState.LYING: "lie",
        PoseState.WALKING: "walk",
        PoseState.FALLING: "fall",
        PoseState.UNKNOWN: "dim",
    }

    def _update_patient_panel(self, rows: list):
        """Cập nhật card BỆNH NHÂN THEO DÕI."""
        w = self._patient_text
        w.config(state="normal")
        w.delete("1.0", "end")
        if not rows:
            w.insert("end", "  Chưa phát hiện bệnh nhân\n", "dim")
        else:
            for name, state, ts in rows:
                label = STATE_LABELS_VN.get(state, "?")
                t_str = time.strftime("%H:%M:%S", time.localtime(ts))
                tag   = self._STATE_TAG.get(state, "dim")
                w.insert("end", f"  {name:<14} ", "dim")
                w.insert("end", f"{label:<8}", tag)
                w.insert("end", f"  {t_str}\n", "dim")
        w.config(state="disabled")

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
