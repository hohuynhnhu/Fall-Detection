"""
Fall Clip Splitter - Tự động cắt video funny falls thành các clip riêng lẻ
Requirements: pip install opencv-python mediapipe scenedetect[opencv] numpy yt-dlp
              ffmpeg phải được cài trên hệ thống
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import subprocess
import json
import os
import re
import time
import cv2
import numpy as np
from pathlib import Path

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False


# ─────────────────────────────────────────────
# YOUTUBE UTILITIES
# ─────────────────────────────────────────────

def is_youtube_url(url: str) -> bool:
    return bool(re.match(
        r'(https?://)?(www\.)?(youtube\.com/watch|youtu\.be/|youtube\.com/shorts/)',
        url.strip()
    ))


def get_video_info(url: str) -> dict:
    """Lấy metadata video YouTube (title, duration, ...)."""
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "uploader": info.get("uploader", ""),
        "id": info.get("id", ""),
    }


def download_youtube_video(url: str, output_dir: str,
                            quality: str = "bestvideo[height<=1080]+bestaudio/best",
                            progress_cb=None) -> str:
    """
    Download video YouTube về máy. Trả về đường dẫn file .mp4.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, "%(title).60s_%(id)s.%(ext)s")

    final_path = [None]

    def _hook(d):
        if d["status"] == "downloading" and progress_cb:
            total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
            downloaded = d.get("downloaded_bytes", 0)
            if total:
                pct = downloaded / total * 100
                speed = d.get("_speed_str", "??")
                eta   = d.get("_eta_str",   "??")
                progress_cb(pct, f"Đang tải: {pct:.0f}%  {speed}  ETA {eta}")
        if d["status"] == "finished":
            final_path[0] = d.get("filename")

    opts = {
        "format": quality,
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_hook],
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        fname = ydl.prepare_filename(info)

    # Resolve final path (có thể đổi extension sau merge)
    p = Path(fname)
    for ext in [".mp4", ".mkv", ".webm"]:
        candidate = p.with_suffix(ext)
        if candidate.exists():
            return str(candidate)
    # Fallback: tìm file mới nhất trong output_dir có id trong tên
    vid_id = info.get("id", "")
    for f in sorted(Path(output_dir).iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if vid_id in f.name and f.suffix in (".mp4", ".mkv", ".webm"):
            return str(f)
    return fname


def get_youtube_stream_url(url: str, quality: str = "best[height<=720]") -> tuple[str, str]:
    """
    Lấy direct stream URL + title để stream trực tiếp vào FFmpeg/OpenCV.
    Trả về (stream_url, title).
    """
    opts = {
        "format": quality + "/bestvideo[height<=720]+bestaudio/best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title", "youtube_video")

    # Ưu tiên format đã merge; nếu không có thì lấy url trực tiếp
    if "url" in info:
        return info["url"], title

    # Tìm format video tốt nhất có cả video+audio
    for fmt in reversed(info.get("formats", [])):
        if (fmt.get("url")
                and fmt.get("vcodec", "none") != "none"
                and fmt.get("acodec", "none") != "none"):
            return fmt["url"], title

    # Fallback: chỉ video (không audio)
    for fmt in reversed(info.get("formats", [])):
        if fmt.get("url") and fmt.get("vcodec", "none") != "none":
            return fmt["url"], title

    raise RuntimeError("Không tìm được stream URL từ video này.")

try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector, AdaptiveDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

QUALITY_OPTIONS = {
    "4K  (2160p)": "bestvideo[height<=2160]+bestaudio/best",
    "FHD (1080p)": "bestvideo[height<=1080]+bestaudio/best",
    "HD  (720p) ": "bestvideo[height<=720]+bestaudio/best",
    "SD  (480p) ": "bestvideo[height<=480]+bestaudio/best",
    "Nhỏ (360p) ": "bestvideo[height<=360]+bestaudio/best",
}


# ─────────────────────────────────────────────
# CORE: Scene detection + fall analysis
# ─────────────────────────────────────────────

def detect_scenes_opencv(video_path: str, threshold: float = 30.0) -> list[dict]:
    """Fallback scene detection using frame difference (no scenedetect required)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    scenes = []
    prev_gray = None
    scene_start = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))
        
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            score = np.mean(diff)
            if score > threshold and frame_idx - scene_start > fps * 1.0:
                scenes.append({
                    "start_frame": scene_start,
                    "end_frame": frame_idx,
                    "start_sec": scene_start / fps,
                    "end_sec": frame_idx / fps,
                    "duration": (frame_idx - scene_start) / fps
                })
                scene_start = frame_idx
        
        prev_gray = gray
        frame_idx += 1
    
    # Last scene
    if frame_idx - scene_start > fps * 0.5:
        scenes.append({
            "start_frame": scene_start,
            "end_frame": frame_idx,
            "start_sec": scene_start / fps,
            "end_sec": frame_idx / fps,
            "duration": (frame_idx - scene_start) / fps
        })
    
    cap.release()
    return scenes


def detect_scenes_scenedetect(video_path: str, threshold: float = 27.0) -> list[dict]:
    """Scene detection using PySceneDetect (more accurate)."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    scenes = []
    for start, end in scene_list:
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        scenes.append({
            "start_frame": start.get_frames(),
            "end_frame": end.get_frames(),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration": end_sec - start_sec
        })
    return scenes


def analyze_scene_for_fall(video_path: str, start_sec: float, end_sec: float,
                             sample_rate: int = 5) -> dict:
    """
    Phân tích scene có người ngã không dựa vào:
    1. Pose landmarks (nếu có MediaPipe)
    2. Chuyển động dọc nhanh (vertical motion)
    3. Aspect ratio của bounding box người
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    
    fall_score = 0.0
    person_detected = False
    body_roi = None  # (x1, y1, x2, y2) normalized
    
    # Motion analysis
    prev_gray = None
    vertical_motions = []
    
    # Person detection via background subtraction heuristic
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50)
    
    frame_count = 0
    analyzed = 0
    
    if MEDIAPIPE_AVAILABLE:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    y_positions = []  # track hip y position over time
    
    while cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 < end_sec:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % sample_rate != 0:
            continue
        analyzed += 1
        
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Background subtraction to find moving person
        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        large_contours = [c for c in contours if cv2.contourArea(c) > 2000]
        if large_contours:
            person_detected = True
            all_pts = np.vstack(large_contours)
            x, y, cw, ch = cv2.boundingRect(all_pts)
            # Expand bounding box 20% for safety
            pad_x = int(cw * 0.2)
            pad_y = int(ch * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)
            body_roi = (x1 / w, y1 / h, x2 / w, y2 / h)
            
            # Aspect ratio: nếu người đang ngã thì width > height
            if ch > 0:
                ar = cw / ch
                if ar > 1.5:  # horizontal = ngã
                    fall_score += 2.0
                elif ar > 1.0:
                    fall_score += 0.5
        
        # MediaPipe pose analysis
        if MEDIAPIPE_AVAILABLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                person_detected = True
                lm = results.pose_landmarks.landmark
                
                # Get key landmarks
                mp_pose_lm = mp.solutions.pose.PoseLandmark
                
                # Hip center y (normalized 0-1, 0=top)
                left_hip = lm[mp_pose_lm.LEFT_HIP]
                right_hip = lm[mp_pose_lm.RIGHT_HIP]
                hip_y = (left_hip.y + right_hip.y) / 2
                y_positions.append(hip_y)
                
                # Shoulder vs hip: nếu shoulder y > hip y => người đang ngã/nằm
                left_shoulder = lm[mp_pose_lm.LEFT_SHOULDER]
                right_shoulder = lm[mp_pose_lm.RIGHT_SHOULDER]
                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                
                # Torso angle: nếu nằm thì shoulder_y gần với hip_y
                torso_vertical_ratio = abs(shoulder_y - hip_y)
                if torso_vertical_ratio < 0.08:  # gần bằng nhau => nằm ngang
                    fall_score += 3.0
                elif torso_vertical_ratio < 0.15:
                    fall_score += 1.0
                
                # Bounding box từ pose
                xs = [lm[i].x for i in range(33) if lm[i].visibility > 0.3]
                ys = [lm[i].y for i in range(33) if lm[i].visibility > 0.3]
                if xs and ys:
                    bx1 = max(0, min(xs) - 0.05)
                    by1 = max(0, min(ys) - 0.05)
                    bx2 = min(1, max(xs) + 0.05)
                    by2 = min(1, max(ys) + 0.05)
                    body_roi = (bx1, by1, bx2, by2)
        
        # Vertical motion detection
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 1, 10, 2, 5, 1.2, 0
            )
            vy = flow[..., 1]
            mean_vy = np.mean(np.abs(vy))
            vertical_motions.append(mean_vy)
        
        prev_gray = gray
    
    # Analyze y_position change (rapid drop = fall)
    if len(y_positions) >= 3:
        y_arr = np.array(y_positions)
        diffs = np.diff(y_arr)
        max_drop = np.max(diffs)  # positive = moving down in image
        if max_drop > 0.15:
            fall_score += 3.0
        elif max_drop > 0.08:
            fall_score += 1.5
    
    # Vertical motion spike
    if vertical_motions:
        max_vm = max(vertical_motions)
        if max_vm > 3.0:
            fall_score += 1.0
    
    cap.release()
    if MEDIAPIPE_AVAILABLE:
        pose.close()
    
    return {
        "fall_score": fall_score,
        "person_detected": person_detected,
        "body_roi": body_roi,  # normalized (x1,y1,x2,y2)
        "analyzed_frames": analyzed
    }


def compute_crop_params(video_path: str, body_roi, padding: float = 0.15):
    """
    Tính toán crop region để focus vào người ngã.
    Đảm bảo thấy đầy đủ tay chân, maintain aspect ratio.
    """
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if body_roi is None:
        return None  # No crop, use full frame
    
    x1n, y1n, x2n, y2n = body_roi
    
    # Add padding
    x1n = max(0, x1n - padding)
    y1n = max(0, y1n - padding)
    x2n = min(1, x2n + padding)
    y2n = min(1, y2n + padding)
    
    # Convert to pixels
    x1 = int(x1n * w)
    y1 = int(y1n * h)
    x2 = int(x2n * w)
    y2 = int(y2n * h)
    
    crop_w = x2 - x1
    crop_h = y2 - y1
    
    # Maintain 9:16 or 16:9 aspect ratio
    target_ar = 9 / 16  # vertical for phone (portrait)
    current_ar = crop_w / crop_h if crop_h > 0 else 1
    
    if current_ar < target_ar:
        # Too tall, expand width
        new_w = int(crop_h * target_ar)
        expand = (new_w - crop_w) // 2
        x1 = max(0, x1 - expand)
        x2 = min(w, x2 + expand)
    else:
        # Too wide, expand height
        new_h = int(crop_w / target_ar)
        expand = (new_h - crop_h) // 2
        y1 = max(0, y1 - expand)
        y2 = min(h, y2 + expand)
    
    # Clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
            "orig_w": w, "orig_h": h}


def cut_clip_ffmpeg(video_path: str, output_path: str,
                    start_sec: float, end_sec: float,
                    crop: dict = None) -> bool:
    """Cắt clip bằng FFmpeg, có thể crop nếu cần."""
    duration = end_sec - start_sec
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
    ]
    
    if crop:
        vf = f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}"
        cmd += ["-vf", vf]
    
    cmd += [
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "23",
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0
    except Exception:
        return False


# ─────────────────────────────────────────────
# GUI Application
# ─────────────────────────────────────────────

class FallClipSplitterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 Fall Clip Splitter")
        self.root.geometry("1100x750")
        self.root.configure(bg="#0f0f13")
        self.root.resizable(True, True)
        
        self.video_path   = tk.StringVar()
        self.output_dir   = tk.StringVar()
        self.threshold    = tk.DoubleVar(value=27.0)
        self.min_duration = tk.DoubleVar(value=2.5)
        self.max_duration = tk.DoubleVar(value=8.0)
        self.fall_threshold = tk.DoubleVar(value=1.0)
        self.enable_crop  = tk.BooleanVar(value=True)
        self.use_pose     = tk.BooleanVar(value=MEDIAPIPE_AVAILABLE)

        # YouTube
        self.yt_url       = tk.StringVar()
        self.yt_mode      = tk.StringVar(value="stream")
        self.yt_quality   = tk.StringVar(value="HD  (720p) ")
        self.yt_info_text = tk.StringVar(value="")

        self.scenes: list[dict] = []
        self.clip_vars: list[dict] = []
        self.processing = False
        
        self._build_ui()
    
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#0f0f13", foreground="#e0e0e0",
                        font=("Consolas", 10))
        style.configure("TFrame", background="#0f0f13")
        style.configure("TLabel", background="#0f0f13", foreground="#e0e0e0")
        style.configure("TButton", background="#1e1e2e", foreground="#cdd6f4",
                        padding=6, relief="flat")
        style.map("TButton",
                  background=[("active", "#313244")],
                  foreground=[("active", "#89dceb")])
        style.configure("Accent.TButton", background="#7c3aed",
                        foreground="white", padding=8)
        style.map("Accent.TButton", background=[("active", "#6d28d9")])
        style.configure("TEntry", fieldbackground="#1e1e2e",
                        foreground="#cdd6f4", insertcolor="#cdd6f4")
        style.configure("TScale", background="#0f0f13", troughcolor="#1e1e2e")
        style.configure("TCheckbutton", background="#0f0f13",
                        foreground="#cdd6f4")
        style.configure("TNotebook", background="#0f0f13", tabmargins=[2,5,2,0])
        style.configure("TNotebook.Tab", background="#1e1e2e",
                        foreground="#bac2de", padding=[12,6])
        style.map("TNotebook.Tab",
                  background=[("selected", "#313244")],
                  foreground=[("selected", "#cdd6f4")])
        style.configure("Treeview", background="#1e1e2e", foreground="#cdd6f4",
                        fieldbackground="#1e1e2e", rowheight=28)
        style.configure("Treeview.Heading", background="#313244",
                        foreground="#89b4fa")
        style.map("Treeview", background=[("selected", "#45475a")])
        style.configure("TProgressbar", troughcolor="#1e1e2e",
                        background="#7c3aed", thickness=8)
        
        # ── Header
        header = tk.Frame(self.root, bg="#0f0f13", pady=12)
        header.pack(fill="x", padx=20)
        tk.Label(header, text="🎬 FALL CLIP SPLITTER",
                 font=("Consolas", 20, "bold"),
                 bg="#0f0f13", fg="#7c3aed").pack(side="left")
        status_badge = tk.Label(
            header,
            text=f"  MediaPipe {'✓' if MEDIAPIPE_AVAILABLE else '✗'}  "
                 f"SceneDetect {'✓' if SCENEDETECT_AVAILABLE else '✗'}  "
                 f"yt-dlp {'✓' if YTDLP_AVAILABLE else '✗'}  ",
            font=("Consolas", 9),
            bg="#1e1e2e", fg="#a6e3a1", pady=4, padx=8
        )
        status_badge.pack(side="right")
        
        # ── Tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        tab0 = ttk.Frame(notebook)
        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)
        notebook.add(tab0, text="  🎥  YouTube  ")
        notebook.add(tab1, text="  ⚙️  Cài đặt & Phân tích  ")
        notebook.add(tab2, text="  ✂️  Quản lý Clips  ")
        
        self._build_youtube_tab(tab0)
        self._build_settings_tab(tab1)
        self._build_clips_tab(tab2)
        self.clips_notebook = notebook
    
    def _build_youtube_tab(self, parent):
        """Tab YouTube: nhập URL, chọn chất lượng, download hoặc stream phân tích."""
        if not YTDLP_AVAILABLE:
            tk.Label(parent,
                     text="⚠️  yt-dlp chưa được cài đặt\n\npip install yt-dlp",
                     bg="#0f0f13", fg="#f38ba8",
                     font=("Consolas", 14), justify="center").pack(expand=True)
            return

        # ── URL input
        self._section(parent, "🎥 URL Video YouTube")
        url_frame = tk.Frame(parent, bg="#0f0f13")
        url_frame.pack(fill="x", padx=20, pady=4)

        url_entry = ttk.Entry(url_frame, textvariable=self.yt_url, font=("Consolas", 10))
        url_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(url_frame, text="🔍 Lấy thông tin",
                   command=self._yt_fetch_info).pack(side="left")

        # ── Info display
        info_lbl = tk.Label(parent, textvariable=self.yt_info_text,
                            bg="#1e1e2e", fg="#89dceb",
                            font=("Consolas", 9), justify="left",
                            anchor="w", padx=10, pady=6, wraplength=700)
        info_lbl.pack(fill="x", padx=20, pady=4)

        # ── Quality selector
        q_frame = tk.Frame(parent, bg="#0f0f13")
        q_frame.pack(fill="x", padx=20, pady=6)
        tk.Label(q_frame, text="Chất lượng:", bg="#0f0f13", fg="#cdd6f4",
                 font=("Consolas", 10)).pack(side="left", padx=(0, 10))
        quality_cb = ttk.Combobox(q_frame, textvariable=self.yt_quality,
                                   values=list(QUALITY_OPTIONS.keys()),
                                   state="readonly", width=18,
                                   font=("Consolas", 10))
        quality_cb.pack(side="left")

        # ── Mode selector
        mode_frame = tk.LabelFrame(parent, text="  Chế độ xử lý  ",
                                    bg="#0f0f13", fg="#89b4fa",
                                    font=("Consolas", 9, "bold"),
                                    bd=1, relief="groove")
        mode_frame.pack(fill="x", padx=20, pady=8)

        # Radio buttons
        r_stream = ttk.Radiobutton(mode_frame, text="⚡  Stream trực tiếp (không download, nhanh hơn)",
                                    variable=self.yt_mode, value="stream")
        r_stream.pack(anchor="w", padx=12, pady=4)
        tk.Label(mode_frame,
                 text="   → Phân tích & cắt clip thẳng từ stream URL, không lưu file gốc",
                 bg="#0f0f13", fg="#6c7086", font=("Consolas", 8)).pack(anchor="w", padx=12)

        r_dl = ttk.Radiobutton(mode_frame, text="⬇️  Download về máy trước rồi xử lý",
                                variable=self.yt_mode, value="download")
        r_dl.pack(anchor="w", padx=12, pady=(8, 4))
        tk.Label(mode_frame,
                 text="   → Tải full video về, sau đó cắt clip (ổn định hơn, cần dung lượng)",
                 bg="#0f0f13", fg="#6c7086", font=("Consolas", 8)).pack(anchor="w", padx=12, pady=(0, 8))

        # ── Output dir cho download
        dl_frame = tk.Frame(parent, bg="#0f0f13")
        dl_frame.pack(fill="x", padx=20, pady=4)
        tk.Label(dl_frame, text="Lưu video tải về vào:", bg="#0f0f13",
                 fg="#bac2de", font=("Consolas", 9)).pack(anchor="w")
        dl_path_row = tk.Frame(dl_frame, bg="#0f0f13")
        dl_path_row.pack(fill="x", pady=2)
        self.yt_dl_dir = tk.StringVar(value=str(Path.home() / "Downloads"))
        ttk.Entry(dl_path_row, textvariable=self.yt_dl_dir, width=55).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(dl_path_row, text="Chọn",
                   command=lambda: self.yt_dl_dir.set(
                       filedialog.askdirectory() or self.yt_dl_dir.get()
                   )).pack(side="left")

        # ── Action button
        btn_row = tk.Frame(parent, bg="#0f0f13")
        btn_row.pack(fill="x", padx=20, pady=14)

        self.yt_action_btn = ttk.Button(
            btn_row,
            text="🚀  Bắt đầu xử lý từ YouTube",
            style="Accent.TButton",
            command=self._yt_start
        )
        self.yt_action_btn.pack(side="left", padx=4)

        ttk.Button(btn_row, text="⬇️  Chỉ Download (không phân tích)",
                   command=self._yt_download_only).pack(side="left", padx=4)

        # ── Progress + log riêng cho YouTube
        self.yt_progress_var = tk.DoubleVar()
        self.yt_progress_lbl = tk.Label(parent, text="",
                                         bg="#0f0f13", fg="#a6e3a1",
                                         font=("Consolas", 9))
        self.yt_progress_lbl.pack(anchor="w", padx=20)
        ttk.Progressbar(parent, variable=self.yt_progress_var,
                        maximum=100).pack(fill="x", padx=20, pady=4)

        # Hint box
        hint = tk.Text(parent, bg="#1e1e2e", fg="#6c7086",
                       font=("Consolas", 8), height=5, relief="flat",
                       state="normal", wrap="word")
        hint.insert("1.0",
            "💡 Hướng dẫn nhanh:\n"
            "  1. Dán URL YouTube vào ô trên (hỗ trợ youtube.com/watch, youtu.be, youtube.com/shorts)\n"
            "  2. Bấm 'Lấy thông tin' để xem tiêu đề & thời lượng\n"
            "  3. Chọn chế độ: Stream (nhanh, không tốn dung lượng) hoặc Download (ổn định hơn)\n"
            "  4. Bấm 'Bắt đầu xử lý' → app sẽ tự chuyển sang tab Phân tích và chạy pipeline"
        )
        hint.configure(state="disabled")
        hint.pack(fill="x", padx=20, pady=6)

    def _yt_fetch_info(self):
        url = self.yt_url.get().strip()
        if not url:
            messagebox.showwarning("Thiếu URL", "Vui lòng nhập URL YouTube!")
            return
        if not is_youtube_url(url):
            messagebox.showwarning("URL không hợp lệ",
                                   "URL không phải YouTube. Hỗ trợ youtube.com và youtu.be")
            return

        self.yt_info_text.set("⏳ Đang lấy thông tin...")

        def _worker():
            try:
                info = get_video_info(url)
                dur = info["duration"]
                m, s = divmod(int(dur), 60)
                h, m = divmod(m, 60)
                dur_str = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
                self.yt_info_text.set(
                    f"✅ {info['title']}\n"
                    f"   Kênh: {info['uploader']}  |  Thời lượng: {dur_str}"
                )
            except Exception as e:
                self.yt_info_text.set(f"❌ Lỗi: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def _yt_set_progress(self, pct, msg=""):
        self.yt_progress_var.set(pct)
        if msg:
            self.yt_progress_lbl.configure(text=msg)
        self.root.update_idletasks()

    def _yt_start(self):
        url = self.yt_url.get().strip()
        if not url:
            messagebox.showwarning("Thiếu URL", "Vui lòng nhập URL YouTube!")
            return
        if not is_youtube_url(url):
            messagebox.showwarning("URL không hợp lệ", "Hỗ trợ youtube.com và youtu.be")
            return
        if self.processing:
            messagebox.showwarning("Đang xử lý", "Vui lòng chờ tác vụ hiện tại xong!")
            return
        if not self.output_dir.get():
            messagebox.showwarning("Chưa chọn thư mục",
                                   "Vui lòng chọn thư mục xuất trong tab Cài đặt!")
            return
        threading.Thread(target=self._yt_worker, args=(url,), daemon=True).start()

    def _yt_worker(self, url):
        self.processing = True
        try:
            mode = self.yt_mode.get()
            quality_key = self.yt_quality.get()
            quality_fmt = QUALITY_OPTIONS.get(quality_key, "bestvideo[height<=720]+bestaudio/best")

            if mode == "download":
                # ── Download mode
                dl_dir = self.yt_dl_dir.get()
                self._yt_set_progress(0, "Đang kết nối YouTube...")
                self._log(f"YouTube: Download video ({quality_key.strip()})...")

                def _dl_progress(pct, msg):
                    self._yt_set_progress(pct * 0.8, msg)
                    self._log(f"  {msg}")

                fpath = download_youtube_video(url, dl_dir, quality_fmt, _dl_progress)
                self._log(f"✅ Đã tải về: {Path(fpath).name}")
                self._yt_set_progress(85, "Đã tải xong, đang nạp vào pipeline...")

                # Set video path và chạy phân tích
                self.root.after(0, lambda: self.video_path.set(fpath))
                if not self.output_dir.get():
                    out = str(Path(fpath).parent / (Path(fpath).stem + "_clips"))
                    self.root.after(0, lambda: self.output_dir.set(out))

                self._yt_set_progress(100, "Chuyển sang phân tích...")
                self.root.after(500, lambda: self.clips_notebook.select(1))
                self.root.after(600, self._start_analyze)

            else:
                # ── Stream mode
                self._yt_set_progress(0, "Đang lấy stream URL...")
                self._log(f"YouTube: Stream trực tiếp ({quality_key.strip()})...")

                stream_url, title = get_youtube_stream_url(url, "best[height<=720]")
                self._log(f"  Stream URL lấy thành công: {title[:60]}...")
                self._yt_set_progress(30, "Đã lấy stream URL, đang phân tích...")

                # Dùng stream URL như file video bình thường
                self.root.after(0, lambda: self.video_path.set(stream_url))
                # Set output dir dựa trên yt_dl_dir
                if not self.output_dir.get():
                    safe_title = re.sub(r'[\/:*?"<>|]', "_", title)[:50]
                    out = str(Path(self.yt_dl_dir.get()) / (safe_title + "_clips"))
                    self.root.after(0, lambda: self.output_dir.set(out))

                self._yt_set_progress(100, "Chuyển sang phân tích...")
                self.root.after(300, lambda: self.clips_notebook.select(1))
                self.root.after(400, self._start_analyze)

        except Exception as e:
            self._log(f"❌ YouTube error: {e}")
            self._yt_set_progress(0, f"Lỗi: {e}")
            messagebox.showerror("Lỗi YouTube", str(e))
        finally:
            self.processing = False

    def _yt_download_only(self):
        url = self.yt_url.get().strip()
        if not url:
            messagebox.showwarning("Thiếu URL", "Vui lòng nhập URL YouTube!")
            return
        if not is_youtube_url(url):
            messagebox.showwarning("URL không hợp lệ", "Hỗ trợ youtube.com và youtu.be")
            return

        quality_key = self.yt_quality.get()
        quality_fmt = QUALITY_OPTIONS.get(quality_key, "bestvideo[height<=720]+bestaudio/best")
        dl_dir = self.yt_dl_dir.get()

        self._yt_set_progress(0, "Bắt đầu tải...")
        self._log(f"Download only: {url}")

        def _worker():
            try:
                def _cb(pct, msg):
                    self._yt_set_progress(pct, msg)
                fpath = download_youtube_video(url, dl_dir, quality_fmt, _cb)
                self._yt_set_progress(100, f"✅ Xong: {Path(fpath).name}")
                self._log(f"✅ Đã tải về: {fpath}")
                self.root.after(0, lambda: messagebox.showinfo(
                    "Tải xong", f"Video đã lưu tại:\n{fpath}"))
            except Exception as e:
                self._log(f"❌ Lỗi tải: {e}")
                self._yt_set_progress(0, f"Lỗi: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def _build_settings_tab(self, parent):
        # Left column: inputs
        left = tk.Frame(parent, bg="#0f0f13")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Input video
        self._section(left, "📁 Video đầu vào")
        row = tk.Frame(left, bg="#0f0f13")
        row.pack(fill="x", pady=4)
        ttk.Entry(row, textvariable=self.video_path, width=50).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row, text="Chọn file",
                   command=self._pick_video).pack(side="left")
        
        # Output dir
        self._section(left, "📂 Thư mục xuất")
        row2 = tk.Frame(left, bg="#0f0f13")
        row2.pack(fill="x", pady=4)
        ttk.Entry(row2, textvariable=self.output_dir, width=50).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row2, text="Chọn thư mục",
                   command=self._pick_output).pack(side="left")
        
        # Settings
        self._section(left, "🔧 Tham số phát hiện")
        
        self._slider(left, "Ngưỡng scene detection",
                     self.threshold, 5, 60,
                     "Giá trị thấp = nhạy hơn (nhiều scene hơn)")
        self._slider(left, "Thời lượng tối thiểu (giây)",
                     self.min_duration, 0.5, 5.0,
                     "Bỏ qua scene ngắn hơn ngưỡng này")
        self._slider(left, "Thời lượng tối đa (giây)",
                     self.max_duration, 3.0, 60.0,
                     "Giới hạn độ dài mỗi clip")
        self._slider(left, "Ngưỡng fall score",
                     self.fall_threshold, 0.0, 5.0,
                     "0 = giữ tất cả scene, cao hơn = chỉ giữ fall rõ")
        
        opt_frame = tk.Frame(left, bg="#0f0f13")
        opt_frame.pack(fill="x", pady=8)
        ttk.Checkbutton(opt_frame, text="Auto-crop focus người ngã",
                        variable=self.enable_crop).pack(side="left", padx=8)
        ttk.Checkbutton(opt_frame,
                        text=f"Dùng MediaPipe Pose {'(không có)' if not MEDIAPIPE_AVAILABLE else ''}",
                        variable=self.use_pose,
                        state="normal" if MEDIAPIPE_AVAILABLE else "disabled"
                        ).pack(side="left", padx=8)
        
        # Analyze button
        btn_row = tk.Frame(left, bg="#0f0f13")
        btn_row.pack(fill="x", pady=12)
        ttk.Button(btn_row, text="🔍  Phân tích Video",
                   style="Accent.TButton",
                   command=self._start_analyze).pack(side="left", padx=4)
        ttk.Button(btn_row, text="✂️  Export tất cả clip đã chọn",
                   command=self._export_all).pack(side="left", padx=4)
        
        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress_label = tk.Label(left, text="",
                                       bg="#0f0f13", fg="#a6e3a1",
                                       font=("Consolas", 9))
        self.progress_label.pack(anchor="w", pady=(4, 0))
        self.progress_bar = ttk.Progressbar(
            left, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=4)
        
        # Right column: log
        right = tk.Frame(parent, bg="#0f0f13", width=300)
        right.pack(side="right", fill="both", padx=(10, 0))
        right.pack_propagate(False)
        
        self._section(right, "📋 Log")
        self.log_text = tk.Text(right, bg="#1e1e2e", fg="#a6e3a1",
                                font=("Consolas", 9), wrap="word",
                                state="disabled", relief="flat",
                                insertbackground="#cdd6f4")
        scroll = ttk.Scrollbar(right, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)
    
    def _build_clips_tab(self, parent):
        # Treeview
        cols = ("sel", "idx", "start", "end", "duration", "score", "status")
        self.tree = ttk.Treeview(parent, columns=cols,
                                  show="headings", selectmode="browse")
        
        self.tree.heading("sel", text="✓")
        self.tree.heading("idx", text="#")
        self.tree.heading("start", text="Bắt đầu")
        self.tree.heading("end", text="Kết thúc")
        self.tree.heading("duration", text="Thời lượng")
        self.tree.heading("score", text="Fall Score")
        self.tree.heading("status", text="Trạng thái")
        
        self.tree.column("sel", width=40, anchor="center")
        self.tree.column("idx", width=40, anchor="center")
        self.tree.column("start", width=90, anchor="center")
        self.tree.column("end", width=90, anchor="center")
        self.tree.column("duration", width=90, anchor="center")
        self.tree.column("score", width=80, anchor="center")
        self.tree.column("status", width=120, anchor="center")
        
        vsb = ttk.Scrollbar(parent, orient="vertical",
                             command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")
        
        # Right panel: clip controls
        ctrl = tk.Frame(parent, bg="#0f0f13", width=260)
        ctrl.pack(side="right", fill="y", padx=12, pady=8)
        ctrl.pack_propagate(False)
        
        self._section(ctrl, "🎬 Chỉnh sửa Clip")

        self.trim_start_var = tk.DoubleVar(value=0.0)
        self.trim_end_var   = tk.DoubleVar(value=0.0)

        def _make_trim_row(parent, label, var):
            """Tạo một hàng trim gồm: label, slider -10→+10 bước 0.1s, nút -/+, hiển thị giá trị."""
            tk.Label(parent, text=label, bg="#0f0f13", fg="#89b4fa",
                     font=("Consolas", 9, "bold")).pack(anchor="w", pady=(10, 2))

            val_row = tk.Frame(parent, bg="#0f0f13")
            val_row.pack(fill="x")

            # Giá trị hiện tại (to giữa)
            val_lbl = tk.Label(val_row, text="0.0 s",
                               bg="#1e1e2e", fg="#f9e2af",
                               font=("Consolas", 11, "bold"),
                               width=8, anchor="center", pady=3)
            val_lbl.pack(side="left", padx=(0, 6))

            def _update_label(*_):
                v = round(var.get(), 1)
                var.set(v)  # snap to 0.1
                color = "#a6e3a1" if v > 0 else ("#f38ba8" if v < 0 else "#f9e2af")
                sign = "+" if v > 0 else ""
                val_lbl.configure(text=f"{sign}{v:.1f} s", fg=color)

            var.trace_add("write", _update_label)

            # Nút -0.1
            btn_minus = tk.Button(val_row, text="−", bg="#313244", fg="#f38ba8",
                                  font=("Consolas", 10, "bold"),
                                  relief="flat", width=2, cursor="hand2",
                                  command=lambda: var.set(round(max(-10, var.get() - 0.1), 1)))
            btn_minus.pack(side="left")

            # Nút +0.1
            btn_plus = tk.Button(val_row, text="+", bg="#313244", fg="#a6e3a1",
                                 font=("Consolas", 10, "bold"),
                                 relief="flat", width=2, cursor="hand2",
                                 command=lambda: var.set(round(min(10, var.get() + 0.1), 1)))
            btn_plus.pack(side="left", padx=(2, 0))

            # Reset
            tk.Button(val_row, text="↺", bg="#313244", fg="#89dceb",
                      font=("Consolas", 10), relief="flat", width=2, cursor="hand2",
                      command=lambda: var.set(0.0)).pack(side="left", padx=(4, 0))

            # Slider
            slider = ttk.Scale(parent, from_=-10.0, to=10.0,
                               variable=var, orient="horizontal")
            slider.pack(fill="x", pady=(4, 0))

            # Snap to 0.1 on mouse release
            def _snap(e):
                var.set(round(var.get(), 1))
            slider.bind("<ButtonRelease-1>", _snap)

            # Tick marks label
            tick_row = tk.Frame(parent, bg="#0f0f13")
            tick_row.pack(fill="x")
            for t, lbl in [("-10", "-10s"), ("-5", "-5s"), ("0", "0"),
                           ("+5", "+5s"), ("+10", "+10s")]:
                tk.Label(tick_row, text=lbl, bg="#0f0f13", fg="#45475a",
                         font=("Consolas", 7)).pack(side="left", expand=True)

        _make_trim_row(ctrl, "◀  Đầu clip (âm = bớt, dương = thêm)", self.trim_start_var)
        _make_trim_row(ctrl, "▶  Cuối clip (âm = bớt, dương = thêm)", self.trim_end_var)

        ttk.Button(ctrl, text="✅  Áp dụng trim",
                   command=self._apply_trim).pack(fill="x", pady=8)
        
        tk.Frame(ctrl, bg="#313244", height=1).pack(fill="x", pady=8)
        
        ttk.Button(ctrl, text="▶️  Preview clip",
                   command=self._preview_clip).pack(fill="x", pady=3)
        ttk.Button(ctrl, text="✂️  Export clip này",
                   command=self._export_single).pack(fill="x", pady=3)
        
        tk.Frame(ctrl, bg="#313244", height=1).pack(fill="x", pady=8)
        
        ttk.Button(ctrl, text="☑️  Chọn tất cả",
                   command=lambda: self._toggle_all(True)).pack(fill="x", pady=2)
        ttk.Button(ctrl, text="☐  Bỏ chọn tất cả",
                   command=lambda: self._toggle_all(False)).pack(fill="x", pady=2)
        ttk.Button(ctrl, text="🗑️  Xóa clip này",
                   command=self._delete_clip).pack(fill="x", pady=6)
        
        tk.Frame(ctrl, bg="#313244", height=1).pack(fill="x", pady=8)
        ttk.Button(ctrl, text="✂️  Export tất cả ✓",
                   style="Accent.TButton",
                   command=self._export_all).pack(fill="x", pady=4)
        
        # Bind tree selection
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Button-1>", self._on_click)
    
    # ── Helpers
    
    def _section(self, parent, title):
        tk.Label(parent, text=title,
                 bg="#0f0f13", fg="#89b4fa",
                 font=("Consolas", 10, "bold")).pack(anchor="w", pady=(12, 4))
    
    def _slider(self, parent, label, var, from_, to, hint=""):
        frame = tk.Frame(parent, bg="#0f0f13")
        frame.pack(fill="x", pady=3)
        
        lbl_frame = tk.Frame(frame, bg="#0f0f13")
        lbl_frame.pack(fill="x")
        tk.Label(lbl_frame, text=label, bg="#0f0f13", fg="#cdd6f4",
                 font=("Consolas", 9)).pack(side="left")
        val_lbl = tk.Label(lbl_frame, textvariable=var,
                           bg="#0f0f13", fg="#f9e2af",
                           font=("Consolas", 9, "bold"), width=6)
        val_lbl.pack(side="right")
        
        ttk.Scale(frame, from_=from_, to=to, variable=var,
                  orient="horizontal").pack(fill="x")
        if hint:
            tk.Label(frame, text=hint, bg="#0f0f13", fg="#6c7086",
                     font=("Consolas", 8)).pack(anchor="w")
    
    def _log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.root.update_idletasks()
    
    def _set_progress(self, pct, msg=""):
        self.progress_var.set(pct)
        if msg:
            self.progress_label.configure(text=msg)
        self.root.update_idletasks()
    
    # ── Actions
    
    def _pick_video(self):
        p = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv"),
                       ("All files", "*.*")])
        if p:
            self.video_path.set(p)
            # Auto set output dir
            if not self.output_dir.get():
                out = str(Path(p).parent / (Path(p).stem + "_clips"))
                self.output_dir.set(out)
            self._log(f"Đã chọn video: {Path(p).name}")
    
    def _pick_output(self):
        p = filedialog.askdirectory()
        if p:
            self.output_dir.set(p)
    
    def _start_analyze(self):
        if not self.video_path.get():
            messagebox.showwarning("Thiếu file", "Vui lòng chọn file video!")
            return
        if self.processing:
            return
        self.processing = True
        threading.Thread(target=self._analyze_worker, daemon=True).start()
    
    def _analyze_worker(self):
        try:
            vpath = self.video_path.get()
            self._log("Bắt đầu phân tích video...")
            self._set_progress(5, "Đang phát hiện scene changes...")
            
            # Scene detection
            if SCENEDETECT_AVAILABLE:
                self._log("Dùng PySceneDetect (AdaptiveDetector)...")
                scenes = detect_scenes_scenedetect(vpath, self.threshold.get())
            else:
                self._log("Dùng OpenCV frame difference (fallback)...")
                scenes = detect_scenes_opencv(vpath, self.threshold.get())
            
            self._log(f"Phát hiện {len(scenes)} scene")
            self._set_progress(20, f"Tìm được {len(scenes)} scene, đang lọc...")
            
            # Filter by duration
            min_d = self.min_duration.get()
            max_d = self.max_duration.get()
            scenes = [s for s in scenes
                      if min_d <= s["duration"] <= max_d]
            self._log(f"Sau khi lọc thời lượng: {len(scenes)} scene")
            
            # Analyze each scene for fall
            self._log("Đang phân tích từng scene...")
            analyzed = []
            for i, scene in enumerate(scenes):
                pct = 20 + (i / len(scenes)) * 60
                self._set_progress(pct,
                    f"Phân tích scene {i+1}/{len(scenes)}...")
                
                result = analyze_scene_for_fall(
                    vpath,
                    scene["start_sec"],
                    scene["end_sec"],
                    sample_rate=3 if self.use_pose.get() else 5
                )
                
                scene.update(result)
                
                # Compute crop
                if self.enable_crop.get() and result["body_roi"]:
                    crop = compute_crop_params(vpath, result["body_roi"])
                    scene["crop"] = crop
                else:
                    scene["crop"] = None
                
                scene["selected"] = result["fall_score"] >= self.fall_threshold.get()
                scene["trim_start"] = 0.0
                scene["trim_end"] = 0.0
                
                analyzed.append(scene)
                self._log(
                    f"  Scene {i+1}: {scene['duration']:.1f}s "
                    f"fall_score={result['fall_score']:.1f} "
                    f"{'✓ GIỮ' if scene['selected'] else '✗ BỎ'}"
                )
            
            self.scenes = analyzed
            self._set_progress(90, "Đang cập nhật UI...")
            self.root.after(0, self._populate_tree)
            self._set_progress(100, f"Xong! {sum(1 for s in analyzed if s['selected'])} clip được chọn")
            self._log(f"✅ Hoàn thành! {sum(1 for s in analyzed if s['selected'])}/{len(analyzed)} clip được chọn")
            
            # Switch to clips tab
            self.root.after(100, lambda: self.clips_notebook.select(1))
            
        except Exception as e:
            self._log(f"❌ Lỗi: {e}")
            import traceback
            self._log(traceback.format_exc())
        finally:
            self.processing = False
    
    def _populate_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.clip_vars.clear()
        
        for i, scene in enumerate(self.scenes):
            sel = "☑" if scene["selected"] else "☐"
            start_str = self._fmt_time(scene["start_sec"])
            end_str = self._fmt_time(scene["end_sec"])
            dur_str = f"{scene['duration']:.1f}s"
            score = f"{scene.get('fall_score', 0):.1f}"
            status = "✓ Chọn" if scene["selected"] else "Bỏ qua"
            
            tag = "selected" if scene["selected"] else "skipped"
            iid = self.tree.insert("", "end",
                values=(sel, i+1, start_str, end_str, dur_str, score, status),
                tags=(tag,))
            scene["_iid"] = iid
        
        self.tree.tag_configure("selected", foreground="#a6e3a1")
        self.tree.tag_configure("skipped", foreground="#6c7086")
    
    def _fmt_time(self, secs):
        m = int(secs // 60)
        s = secs % 60
        return f"{m:02d}:{s:05.2f}"
    
    def _get_selected_scene(self):
        sel = self.tree.selection()
        if not sel:
            return None, None
        iid = sel[0]
        for i, s in enumerate(self.scenes):
            if s.get("_iid") == iid:
                return i, s
        return None, None
    
    def _on_select(self, event):
        idx, scene = self._get_selected_scene()
        if scene:
            self.trim_start_var.set(scene.get("trim_start", 0.0))
            self.trim_end_var.set(scene.get("trim_end", 0.0))
    
    def _on_click(self, event):
        """Toggle checkbox column."""
        region = self.tree.identify_region(event.x, event.y)
        col = self.tree.identify_column(event.x)
        if region == "cell" and col == "#1":
            iid = self.tree.identify_row(event.y)
            if iid:
                for scene in self.scenes:
                    if scene.get("_iid") == iid:
                        scene["selected"] = not scene["selected"]
                        sel = "☑" if scene["selected"] else "☐"
                        status = "✓ Chọn" if scene["selected"] else "Bỏ qua"
                        tag = "selected" if scene["selected"] else "skipped"
                        self.tree.item(iid, values=(
                            sel,
                            self.tree.item(iid)["values"][1],
                            self.tree.item(iid)["values"][2],
                            self.tree.item(iid)["values"][3],
                            self.tree.item(iid)["values"][4],
                            self.tree.item(iid)["values"][5],
                            status
                        ), tags=(tag,))
                        break
    
    def _apply_trim(self):
        idx, scene = self._get_selected_scene()
        if scene is None:
            return
        
        ts = self.trim_start_var.get()
        te = self.trim_end_var.get()
        scene["trim_start"] = ts
        scene["trim_end"] = te
        
        new_start = scene["start_sec"] + ts
        new_end = scene["end_sec"] + te
        new_dur = new_end - new_start
        
        iid = scene["_iid"]
        vals = list(self.tree.item(iid)["values"])
        vals[2] = self._fmt_time(max(0, new_start))
        vals[3] = self._fmt_time(new_end)
        vals[4] = f"{new_dur:.1f}s"
        self.tree.item(iid, values=vals)
        self._log(f"Clip #{idx+1}: trim đầu {ts:+.1f}s, cuối {te:+.1f}s")
    
    def _preview_clip(self):
        idx, scene = self._get_selected_scene()
        if scene is None:
            messagebox.showinfo("Chưa chọn", "Vui lòng chọn clip!")
            return
        
        vpath = self.video_path.get()
        start = max(0, scene["start_sec"] + scene.get("trim_start", 0))
        end = scene["end_sec"] + scene.get("trim_end", 0)
        
        self._log(f"Preview clip #{idx+1}: {self._fmt_time(start)} → {self._fmt_time(end)}")
        
        # Play with OpenCV in a thread
        threading.Thread(
            target=self._play_preview,
            args=(vpath, start, end, scene.get("crop") if self.enable_crop.get() else None),
            daemon=True
        ).start()
    
    def _play_preview(self, vpath, start, end, crop):
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cv2.namedWindow("Preview - nhấn Q để thoát", cv2.WINDOW_NORMAL)
        
        # Compute crop pixels
        if crop:
            cx, cy, cw, ch = crop["x"], crop["y"], crop["w"], crop["h"]
        
        delay = max(1, int(1000 / fps))
        
        while cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 < end:
            ret, frame = cap.read()
            if not ret:
                break
            
            if crop:
                frame = frame[cy:cy+ch, cx:cx+cw]
            
            # Overlay info
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            cv2.putText(frame, f"Time: {self._fmt_time(ts)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 100), 2)
            
            cv2.imshow("Preview - nhấn Q để thoát", frame)
            key = cv2.waitKey(delay)
            if key in (ord('q'), ord('Q'), 27):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _export_single(self):
        idx, scene = self._get_selected_scene()
        if scene is None:
            messagebox.showinfo("Chưa chọn", "Vui lòng chọn clip!")
            return
        threading.Thread(
            target=self._export_worker, args=([scene], [idx]),
            daemon=True
        ).start()
    
    def _export_all(self):
        selected = [(i, s) for i, s in enumerate(self.scenes) if s.get("selected")]
        if not selected:
            messagebox.showwarning("Không có clip", "Không có clip nào được chọn!")
            return
        idxs, scenes = zip(*selected)
        threading.Thread(
            target=self._export_worker, args=(list(scenes), list(idxs)),
            daemon=True
        ).start()
    
    def _export_worker(self, scenes_to_export, indices):
        vpath = self.video_path.get()
        out_dir = self.output_dir.get()
        
        if not out_dir:
            messagebox.showwarning("Thiếu thư mục", "Vui lòng chọn thư mục xuất!")
            return
        
        os.makedirs(out_dir, exist_ok=True)
        self._log(f"Xuất {len(scenes_to_export)} clip vào: {out_dir}")
        
        for i, (scene, idx) in enumerate(zip(scenes_to_export, indices)):
            pct = (i / len(scenes_to_export)) * 100
            self._set_progress(pct, f"Xuất clip {i+1}/{len(scenes_to_export)}...")
            
            start = max(0, scene["start_sec"] + scene.get("trim_start", 0))
            end = scene["end_sec"] + scene.get("trim_end", 0)
            
            out_name = f"fall_clip_{idx+1:03d}_{self._fmt_time(start).replace(':', '-')}.mp4"
            out_path = os.path.join(out_dir, out_name)
            
            crop = scene.get("crop") if self.enable_crop.get() else None
            
            ok = cut_clip_ffmpeg(vpath, out_path, start, end, crop)
            
            status = "✅ OK" if ok else "❌ Lỗi"
            self._log(f"  Clip #{idx+1}: {status} → {out_name}")
            
            # Update tree
            iid = scene.get("_iid")
            if iid:
                vals = list(self.tree.item(iid)["values"])
                vals[6] = "Đã xuất ✅" if ok else "Lỗi ❌"
                self.root.after(0, lambda i=iid, v=vals: self.tree.item(i, values=v))
        
        self._set_progress(100, f"Xuất xong {len(scenes_to_export)} clip!")
        self._log(f"✅ Hoàn thành! Kiểm tra thư mục: {out_dir}")
        self.root.after(0, lambda: messagebox.showinfo(
            "Xuất xong",
            f"Đã xuất {len(scenes_to_export)} clip\nvào: {out_dir}"
        ))
    
    def _toggle_all(self, state):
        for scene in self.scenes:
            scene["selected"] = state
            iid = scene.get("_iid")
            if iid:
                sel = "☑" if state else "☐"
                status = "✓ Chọn" if state else "Bỏ qua"
                tag = "selected" if state else "skipped"
                vals = list(self.tree.item(iid)["values"])
                vals[0] = sel
                vals[6] = status
                self.tree.item(iid, values=vals, tags=(tag,))
    
    def _delete_clip(self):
        idx, scene = self._get_selected_scene()
        if scene is None:
            return
        iid = scene.get("_iid")
        self.tree.delete(iid)
        self.scenes.pop(idx)
        self._log(f"Đã xóa clip #{idx+1}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = FallClipSplitterApp(root)
    root.mainloop()