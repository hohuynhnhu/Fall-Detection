"""
test_model_realtime.py
Test model Transformer với camera realtime
- Phát tiếng beep khi phát hiện fall
- Kiểm tra pose + velocity trước khi infer
- Q=thoát  R=reset  S=screenshot
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import time
import threading
from collections import deque
from datetime import datetime

import mediapipe as mp
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
# ── NGƯỠNG QUYẾT ĐỊNH FALL ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

FALL_THR       = 0.7    # ★ Ngưỡng 1: P(fall) >= 0.6 → kết luận là fall
                         #   Tăng lên 0.7-0.8 nếu báo nhầm nhiều
                         #   Giảm xuống 0.5 nếu bỏ sót fall thật

MIN_POSE_RATIO = 0.5    # ★ Ngưỡng 2: >= 50% frames trong buffer phải
                         #   detect được người → mới đưa vào model
                         #   Tránh infer khi không có người trong frame

MAX_VEL_STILL  = 0.008  # ★ Ngưỡng 3: velocity < 0.008 → người đang nằm im
                         #   → không thể là fall → bỏ qua
                         #   Tăng lên 0.01-0.015 nếu vẫn báo nhầm khi nằm im
                         #   Giảm xuống 0.005 nếu bỏ sót fall chuyển động chậm

BEEP_COOLDOWN  = 3.0    # ★ Cooldown giữa 2 lần beep (giây)
                         #   Tránh beep liên tục khi fall kéo dài

# ══════════════════════════════════════════════════════════════════════════════

CKPT_PATH    = r"D:\luan_van\Fall-Detection\train_tranformer\dataset_1\checkpoints1\best_model.pth"
CAMERA_INDEX = 0
FLIP         = True
DISPLAY_WIDTH = 960     # chiều ngang cửa sổ hiển thị

# ── Âm thanh cảnh báo ─────────────────────────────────────────────────────────

def beep_alert():
    """Phát 3 tiếng beep cảnh báo trong thread riêng — không block camera"""
    def _play():
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1000, 300)
                time.sleep(0.1)
        except Exception:
            # Fallback nếu không có winsound (Linux/Mac)
            try:
                os.system("echo '\a'")
            except Exception:
                pass
    threading.Thread(target=_play, daemon=True).start()

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
        else:
            mp_kp = np.zeros((33, 4), dtype=np.float32)

        yr = self.yolo(frame, verbose=False)
        if yr[0].keypoints is not None and len(yr[0].keypoints.data) > 0:
            kp = yr[0].keypoints.data[0].cpu().numpy()
            yolo_kp = np.zeros((17, 4), dtype=np.float32)
            for i, k in enumerate(kp[:17]):
                yolo_kp[i] = [k[0]/w, k[1]/h, 0.0, k[2]]
        else:
            yolo_kp = np.zeros((17, 4), dtype=np.float32)

        detected = not np.all(mp_kp == 0) or not np.all(yolo_kp == 0)
        raw_kp   = np.concatenate([mp_kp, yolo_kp]).flatten()
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
    """Max velocity Y của center of mass trong buffer"""
    y_coords = buf_arr[:, 1::4]       # tất cả tọa độ y (index 1,5,9,...)
    com_y    = y_coords.mean(axis=1)  # center of mass Y mỗi frame
    vel      = np.abs(np.diff(com_y)) # velocity frame-to-frame
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
                 fps, detected, fall_count, reason, max_vel):
    h, w    = frame.shape[:2]
    is_fall = pred == 1 and p_fall >= FALL_THR

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 80), (12, 12, 22), -1)

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
    cv2.putText(frame, label, (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

    # P(fall) bar — NGƯỠNG 1
    bar_x = 300
    bar_w = w - bar_x - 140
    filled = int(bar_w * p_fall)
    bar_c  = (50, 50, 255) if p_fall >= FALL_THR else (50, 220, 80)
    cv2.rectangle(frame, (bar_x, 16), (bar_x+bar_w, 36), (40,40,60), -1)
    if filled > 0:
        cv2.rectangle(frame, (bar_x, 16), (bar_x+filled, 36), bar_c, -1)
    # Vạch ngưỡng FALL_THR
    thr_x = bar_x + int(bar_w * FALL_THR)
    cv2.line(frame, (thr_x, 12), (thr_x, 40), (255, 220, 50), 2)
    cv2.putText(frame, f"thr={FALL_THR}", (thr_x-20, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 50), 1)
    cv2.putText(frame, f"P(fall)={p_fall:.2f}",
                (w-132, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200,200,220), 1)

    # Buffer progress
    bfill = int(220 * buf_size / num_frames)
    cv2.rectangle(frame, (12, 50), (232, 60), (30,30,50), -1)
    cv2.rectangle(frame, (12, 50), (12+bfill, 60), (100,180,255), -1)

    # Info line — NGƯỠNG 2 + 3
    vel_color = (180,100,100) if max_vel > MAX_VEL_STILL else (100,180,100)
    info = (f"Buffer {buf_size}/{num_frames}  "
            f"FPS {fps:.0f}  "
            f"Pose {'OK' if detected else '--'}  "
            f"vel={max_vel:.4f}(thr={MAX_VEL_STILL})")
    cv2.putText(frame, info, (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, vel_color, 1)

    # Reason + fall count
    cv2.putText(frame, f"[{reason}]  Falls: {fall_count}",
                (w-260, h-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (180,100,100) if fall_count > 0 else (100,100,130), 1)

    # Shortcuts
    cv2.putText(frame, "Q=quit  R=reset  S=screenshot",
                (12, h-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (80,80,100), 1)

    # Fall alert border
    if is_fall:
        cv2.rectangle(frame, (0,0), (w-1,h-1), (50,50,255), 5)

    return frame

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"\nNGƯỠNG QUYẾT ĐỊNH:")
    print(f"  FALL_THR       = {FALL_THR}    ← P(fall) >= {FALL_THR} → FALL")
    print(f"  MIN_POSE_RATIO = {MIN_POSE_RATIO}   ← cần detect người >= {MIN_POSE_RATIO:.0%} frames")
    print(f"  MAX_VEL_STILL  = {MAX_VEL_STILL} ← velocity < {MAX_VEL_STILL} → nằm im → SAFE\n")

    model, num_frames, input_dim = load_model(CKPT_PATH)
    model     = model.to(device)
    extractor = PoseExtractor()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Không mở được camera {CAMERA_INDEX}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"Camera {CAMERA_INDEX} opened.")
    print("Q=thoát  R=reset  S=screenshot\n")

    buffer        = deque(maxlen=num_frames)
    predict_every = max(1, num_frames // 4)
    frame_count   = 0
    p_fall, pred  = 0.0, 0
    fall_count    = 0
    last_fall     = False
    last_beep_t   = 0.0
    reason        = "warming up"
    max_vel       = 0.0

    fps_display = 0.0
    t0          = time.time()
    fps_cnt     = 0

    os.makedirs("screenshots", exist_ok=True)
    cv2.namedWindow("Fall Detection — Realtime", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Mất kết nối camera.")
            break

        if FLIP:
            frame = cv2.flip(frame, 1)

        frame_count += 1
        fps_cnt     += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps_display = fps_cnt / (now - t0)
            fps_cnt = 0; t0 = now

        raw_kp, frame, detected = extractor.extract(frame)
        buffer.append(raw_kp)

        # ── Quyết định fall ──────────────────────────────────────────────────
        if (len(buffer) == num_frames and
                frame_count % predict_every == 0):

            buf_arr    = np.array(buffer, dtype=np.float32)
            mp_part    = buf_arr[:, :132]
            pose_ratio = np.any(np.abs(mp_part) > 1e-3, axis=1).mean()
            max_vel    = check_velocity(buf_arr)

            # NGƯỠNG 2: kiểm tra có người không
            if pose_ratio < MIN_POSE_RATIO:
                p_fall, pred = 0.0, 0
                reason = f"no_pose {pose_ratio:.0%}"

            # NGƯỠNG 3: kiểm tra có chuyển động không
            elif max_vel < MAX_VEL_STILL:
                p_fall, pred = 0.0, 0
                reason = f"still"

            else:
                # NGƯỠNG 1: đưa vào model
                p_fall, pred = infer(model, buf_arr, device)
                is_fall = pred == 1 and p_fall >= FALL_THR
                reason  = f"fall" if is_fall else "safe"

                # Đếm + beep khi fall mới xảy ra (cạnh lên)
                if is_fall and not last_fall:
                    fall_count += 1
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"  [{ts}] FALL DETECTED! "
                          f"P={p_fall:.3f}  vel={max_vel:.4f}  "
                          f"(event #{fall_count})")

                    # Phát beep nếu chưa qua cooldown
                    if now - last_beep_t >= BEEP_COOLDOWN:
                        beep_alert()
                        last_beep_t = now

            last_fall = (pred == 1 and p_fall >= FALL_THR)

        frame = draw_overlay(frame, p_fall, pred,
                             len(buffer), num_frames,
                             fps_display, detected,
                             fall_count, reason, max_vel)

        # Resize hiển thị
        dh, dw = frame.shape[:2]
        scale  = DISPLAY_WIDTH / max(dw, 1)
        disp_h = int(dh * scale)
        display = cv2.resize(frame, (DISPLAY_WIDTH, disp_h),
                             interpolation=cv2.INTER_AREA)
        cv2.resizeWindow("Fall Detection — Realtime", DISPLAY_WIDTH, disp_h)
        cv2.imshow("Fall Detection — Realtime", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            buffer.clear()
            p_fall, pred = 0.0, 0
            reason = "reset"
            print("Buffer reset.")
        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"screenshots/realtime_{ts}.jpg"
            cv2.imwrite(path, frame)
            print(f"Screenshot: {path}")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"\nKết thúc. Tổng fall events: {fall_count}")


if __name__ == "__main__":
    main()