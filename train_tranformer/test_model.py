"""
test_model_video.py
Test model Transformer với video file
- Chọn video qua file dialog
- Kiểm tra pose detected trước khi infer
- Kiểm tra velocity (nằm im → không fall)
- Hiển thị cửa sổ 640px
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import time
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import mediapipe as mp
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────

CKPT_PATH      = r"D:\luan_van\Fall-Detection\train_tranformer\dataset_1\checkpoints1\best_model.pth"
VIDEO_PATH     = r""    # để trống → mở file dialog
FALL_THR       = 0.6    # ngưỡng P(fall)
MIN_POSE_RATIO = 0.5    # >= 50% frames phải có người
MAX_VEL_STILL  = 0.008  # velocity thấp hơn này → nằm im → không fall
DISPLAY_WIDTH  = 640    # chiều ngang cửa sổ hiển thị

# ── Model ─────────────────────────────────────────────────────────────────────

class PoseTransformer(nn.Module):
    def __init__(self, input_dim, num_frames,
                 d_model=128, nhead=4, num_layers=2,
                 dropout=0.4, num_classes=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model), nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.pos_emb   = nn.Parameter(torch.randn(1, num_frames+1, d_model)*0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)*0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Dropout(dropout*0.5), nn.Linear(32, num_classes),
        )

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_emb
        return self.head(self.transformer(x)[:, 0])

# ── Load model ────────────────────────────────────────────────────────────────

def load_model(ckpt_path):
    ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg        = ckpt["config"]
    num_frames = int(cfg["num_frames"])
    input_dim  = int(cfg["input_dim"])
    model = PoseTransformer(
        input_dim=input_dim, num_frames=num_frames,
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.4)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model: num_frames={num_frames}  input_dim={input_dim}  "
          f"kfold_mean={ckpt.get('kfold_mean',0):.1f}%")
    return model, num_frames, input_dim

# ── Pose extractor ────────────────────────────────────────────────────────────

class PoseExtractor:
    def __init__(self):
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.yolo    = YOLO("yolov8n-pose.pt")
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp_pose

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < 80:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            frame = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]),
                                  cv2.COLOR_LAB2BGR)
        return frame

    def extract(self, frame):
        h, w  = frame.shape[:2]
        frame = self.preprocess(frame)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe
        res = self.pose.process(rgb)
        if res.pose_landmarks:
            mp_kp = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                               for lm in res.pose_landmarks.landmark],
                              dtype=np.float32)
            self.mp_draw.draw_landmarks(
                frame, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(80,220,80), thickness=2,
                                          circle_radius=3),
                self.mp_draw.DrawingSpec(color=(120,255,120), thickness=1),
            )
            detected = True
        else:
            mp_kp    = np.zeros((33, 4), dtype=np.float32)
            detected = False

        # YOLO
        yr = self.yolo(frame, verbose=False)
        if yr[0].keypoints is not None and len(yr[0].keypoints.data) > 0:
            kp = yr[0].keypoints.data[0].cpu().numpy()
            yolo_kp = np.zeros((17, 4), dtype=np.float32)
            for i, k in enumerate(kp[:17]):
                yolo_kp[i] = [k[0]/w, k[1]/h, 0.0, k[2]]
            detected = True
        else:
            yolo_kp = np.zeros((17, 4), dtype=np.float32)

        raw_kp = np.concatenate([mp_kp, yolo_kp]).flatten()  # (200,)
        return raw_kp, frame, detected

    def close(self):
        self.pose.close()

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_sequence(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-6, 1.0, std)
    return (arr - mean) / std

def check_velocity(buf_arr: np.ndarray) -> float:
    """
    Tính max velocity trục Y của center of mass trong buffer
    Nằm im  → velocity ≈ 0
    Té ngã  → velocity cao đột ngột
    """
    # Lấy tọa độ Y của tất cả keypoints (index 1, 5, 9, ... trong vector 200)
    y_coords = buf_arr[:, 1::4]          # (T, 50) — tọa độ y của 50 keypoints
    com_y    = y_coords.mean(axis=1)     # (T,) — center of mass Y mỗi frame
    vel      = np.abs(np.diff(com_y))    # (T-1,) — velocity frame-to-frame
    return float(vel.max()) if len(vel) > 0 else 0.0

def infer(model, buf_arr, device):
    X = normalize_sequence(buf_arr.copy())
    t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)
        probs  = torch.softmax(logits, dim=1)[0]
    return float(probs[1]), int(logits.argmax(1).item())

# ── Draw overlay ──────────────────────────────────────────────────────────────

def draw_overlay(frame, p_fall, pred, buf_size, num_frames,
                 fps, reason=""):
    h, w    = frame.shape[:2]
    is_fall = pred == 1 and p_fall >= FALL_THR

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 70), (12, 12, 22), -1)

    # Status
    if buf_size < num_frames:
        label = f"Warming up... {buf_size}/{num_frames}"
        color = (180, 180, 60)
    elif is_fall:
        label = "FALL DETECTED!"
        color = (50, 50, 255)
    else:
        label = "SAFE"
        color = (50, 220, 80)

    cv2.putText(frame, label, (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # P(fall) bar
    bar_x = 280
    bar_w = w - bar_x - 130
    filled = int(bar_w * p_fall)
    bar_c  = (50, 50, 255) if p_fall >= FALL_THR else (50, 220, 80)
    cv2.rectangle(frame, (bar_x, 16), (bar_x + bar_w, 36), (40, 40, 60), -1)
    if filled > 0:
        cv2.rectangle(frame, (bar_x, 16), (bar_x + filled, 36), bar_c, -1)
    cv2.putText(frame, f"P(fall) {p_fall:.2f}",
                (w - 122, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 220), 1)

    # Buffer bar
    bfill = int(220 * buf_size / num_frames)
    cv2.rectangle(frame, (12, 48), (232, 58), (30, 30, 50), -1)
    cv2.rectangle(frame, (12, 48), (12 + bfill, 58), (100, 180, 255), -1)

    # Info line
    info = f"Buffer {buf_size}/{num_frames}  FPS {fps:.0f}"
    if reason:
        info += f"  [{reason}]"
    cv2.putText(frame, info, (12, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 140), 1)

    # Fall border
    if is_fall:
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (50, 50, 255), 5)

    return frame

# ── File dialog ───────────────────────────────────────────────────────────────

def choose_video() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Chọn video để test Fall Detection",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("All files",   "*.*"),
        ],
    )
    root.destroy()
    return path or ""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Chọn video
    video_path = VIDEO_PATH or choose_video()
    if not video_path:
        print("Không có video được chọn. Thoát.")
        sys.exit(0)
    if not os.path.exists(video_path):
        print(f"ERROR: {video_path}")
        sys.exit(1)

    print(f"Video: {video_path}")

    # Load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, num_frames, input_dim = load_model(CKPT_PATH)
    model     = model.to(device)
    extractor = PoseExtractor()

    cap   = cv2.VideoCapture(video_path)
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frames: {total}  FPS: {fps_v:.0f}")
    print("Q = thoát  |  SPACE = pause\n")

    buffer        = deque(maxlen=num_frames)
    predict_every = max(1, num_frames // 4)
    frame_count   = 0
    p_fall, pred  = 0.0, 0
    reason        = ""
    paused        = False
    fps_display   = 0.0
    t0            = time.time()
    fps_cnt       = 0

    cv2.namedWindow("Fall Detection — Test Video", cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Video kết thúc.")
                break

            frame_count += 1
            fps_cnt     += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps_display = fps_cnt / (now - t0)
                fps_cnt = 0; t0 = now

            # Extract keypoints
            raw_kp, frame, detected = extractor.extract(frame)
            buffer.append(raw_kp)

            # Infer khi đủ buffer
            if (len(buffer) == num_frames and
                    frame_count % predict_every == 0):

                buf_arr = np.array(buffer, dtype=np.float32)

                # Kiểm tra 1: có detect được người không
                mp_part   = buf_arr[:, :132]
                pose_ratio = np.any(np.abs(mp_part) > 1e-3, axis=1).mean()

                if pose_ratio < MIN_POSE_RATIO:
                    p_fall, pred = 0.0, 0
                    reason = f"No pose {pose_ratio:.0%}"

                else:
                    # Kiểm tra 2: người có đang di chuyển không
                    max_vel = check_velocity(buf_arr)

                    if max_vel < MAX_VEL_STILL:
                        # Nằm im → không thể là fall
                        p_fall, pred = 0.0, 0
                        reason = f"Still vel={max_vel:.4f}"
                    else:
                        # Có chuyển động → để model quyết định
                        p_fall, pred = infer(model, buf_arr, device)
                        reason = f"vel={max_vel:.4f}"

                # Log
                if pred == 1 and p_fall >= FALL_THR:
                    print(f"  [Frame {frame_count:>5}] FALL  "
                          f"P={p_fall:.3f}  {reason}")

            frame = draw_overlay(frame, p_fall, pred,
                                 len(buffer), num_frames,
                                 fps_display, reason)

        # Resize hiển thị
        dh, dw = frame.shape[:2]
        scale  = DISPLAY_WIDTH / max(dw, 1)
        disp_w = DISPLAY_WIDTH
        disp_h = int(dh * scale)
        display = cv2.resize(frame, (disp_w, disp_h),
                             interpolation=cv2.INTER_AREA)
        cv2.resizeWindow("Fall Detection — Test Video", disp_w, disp_h)
        cv2.imshow("Fall Detection — Test Video", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("Done.")


if __name__ == "__main__":
    main()