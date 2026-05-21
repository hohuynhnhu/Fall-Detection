"""
extract_keypoints.py
Trích xuất keypoints từ video dùng MediaPipe + YOLO Pose (fusion)
Output: X.npy (N, num_frames, 200), y.npy (N,), config.json
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_DIR = r"D:\luan_van\Fall-Detection\train_tranformer\dataset_1"
OUTPUT_DIR  = r"D:\luan_van\Fall-Detection\train_tranformer\dataset_1"
LABELS      = {"fall": 1, "non_fall": 0}

NUM_FRAMES  = 30
WINDOW_SEC  = 3.0

# Threshold skip: bỏ qua nếu > X% frames không detect được người
# Tăng lên 0.7 để giảm số video bị skip
SKIP_THRESHOLD = 0.7

INPUT_DIM   = 200
MP_KP       = 33
YOLO_KP     = 17

# ── Init models ───────────────────────────────────────────────────────────────

print("Loading MediaPipe...")
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.3,   # giảm từ 0.5 → 0.3 để detect tốt hơn khi tối
    min_tracking_confidence=0.3,
)

print("Loading YOLO Pose...")
yolo_model = YOLO("yolov8n-pose.pt")
print("Models ready.\n")

# ── Analyze dataset ───────────────────────────────────────────────────────────

def analyze_dataset(dataset_dir: str):
    all_frames, all_fps = [], []
    label_stats = {}

    for label in ["fall", "non_fall"]:
        folder = os.path.join(dataset_dir, label)
        if not os.path.exists(folder):
            print(f"  WARNING: folder không tồn tại: {folder}")
            continue
        videos = (list(Path(folder).glob("*.mp4")) +
                  list(Path(folder).glob("*.avi")) +
                  list(Path(folder).glob("*.mov")))

        durations = []
        for vp in videos:
            cap   = cv2.VideoCapture(str(vp))
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total > 0:
                all_frames.append(total)
                all_fps.append(fps)
                durations.append(total / fps)

        if durations:
            label_stats[label] = {
                "count":    len(durations),
                "min_dur":  min(durations),
                "max_dur":  max(durations),
                "mean_dur": np.mean(durations),
            }

    print(f"\n{'='*50}")
    print(f"DATASET ANALYSIS")
    print(f"  Tổng video: {len(all_frames)}")
    for label, s in label_stats.items():
        print(f"\n  [{label}] {s['count']} videos")
        print(f"    Duration: {s['min_dur']:.1f}s → {s['max_dur']:.1f}s "
              f"(TB {s['mean_dur']:.1f}s)")
    print(f"\n  NUM_FRAMES    : {NUM_FRAMES}")
    print(f"  SKIP_THRESHOLD: >{SKIP_THRESHOLD:.0%} frames zero → skip")
    print(f"  fall     → lấy toàn bộ video")
    print(f"  non_fall → lấy {WINDOW_SEC}s cuối")
    print(f"{'='*50}\n")

# ── Preprocess frame ──────────────────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_bright = np.mean(gray)
    if mean_bright < 80:
        lab        = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced   = cv2.merge([l_enhanced, a, b])
        frame      = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return frame

# ── Interpolate zero frames ───────────────────────────────────────────────────

def interpolate_missing(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    for i in range(arr.shape[0]):
        if not np.all(arr[i] == 0):
            continue
        prev = next((arr[j] for j in range(i - 1, -1, -1)
                     if not np.all(arr[j] == 0)), None)
        nxt  = next((arr[j] for j in range(i + 1, arr.shape[0])
                     if not np.all(arr[j] == 0)), None)
        if prev is not None and nxt is not None:
            arr[i] = (prev + nxt) / 2.0
        elif prev is not None:
            arr[i] = prev
        elif nxt is not None:
            arr[i] = nxt
    return arr

# ── Normalize sequence ────────────────────────────────────────────────────────

def normalize_sequence(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-6, 1.0, std)
    return (arr - mean) / std

# ── Extract 1 frame ───────────────────────────────────────────────────────────

def extract_frame_fusion(frame: np.ndarray, h: int, w: int) -> np.ndarray:
    frame = preprocess_frame(frame)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe
    mp_result = pose.process(rgb)
    if mp_result.pose_landmarks:
        mp_kp = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in mp_result.pose_landmarks.landmark
        ], dtype=np.float32)
    else:
        mp_kp = np.zeros((MP_KP, 4), dtype=np.float32)

    # YOLO Pose
    yolo_result = yolo_model(frame, verbose=False)
    if (yolo_result[0].keypoints is not None and
            len(yolo_result[0].keypoints.data) > 0):
        kp_data = yolo_result[0].keypoints.data[0].cpu().numpy()
        yolo_kp = np.zeros((YOLO_KP, 4), dtype=np.float32)
        for i, k in enumerate(kp_data[:YOLO_KP]):
            yolo_kp[i] = [k[0] / w, k[1] / h, 0.0, k[2]]
    else:
        yolo_kp = np.zeros((YOLO_KP, 4), dtype=np.float32)

    combined = np.concatenate([mp_kp, yolo_kp], axis=0)
    return combined.flatten()

# ── Extract 1 video ───────────────────────────────────────────────────────────

def extract_one(video_path: str, label: str) -> np.ndarray | None:
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    name  = Path(video_path).name

    # Tiêu chí 1: Video không đọc được
    if total < 2 or h == 0 or w == 0:
        print(f"    SKIP [video lỗi] {name} — frames={total} h={h} w={w}")
        cap.release()
        return None

    # Xác định window
    if label == "fall":
        start = 0
        end   = total - 1
    else:
        window_frames = int(WINDOW_SEC * fps)
        start = max(0, total - window_frames)
        end   = total - 1

    indices = np.linspace(start, end, NUM_FRAMES, dtype=int)
    seq     = []
    mp_ok   = 0  # đếm frame MediaPipe detect được
    yolo_ok = 0  # đếm frame YOLO detect được

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            seq.append(np.zeros(INPUT_DIM, dtype=np.float32))
            continue

        # Đếm riêng để debug
        frame_pre = preprocess_frame(frame)
        rgb = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2RGB)
        mp_res = pose.process(rgb)
        if mp_res.pose_landmarks:
            mp_ok += 1

        yr = yolo_model(frame_pre, verbose=False)
        if yr[0].keypoints is not None and len(yr[0].keypoints.data) > 0:
            yolo_ok += 1

        kp = extract_frame_fusion(frame, h, w)
        seq.append(kp)

    cap.release()
    arr = np.array(seq, dtype=np.float32)

    # Tiêu chí 2: Quá nhiều frame không detect được
    zero_frames = np.all(arr == 0, axis=1).sum()
    zero_ratio  = zero_frames / NUM_FRAMES

    # Log chi tiết từng video
    status = "OK" if zero_ratio <= SKIP_THRESHOLD else f"SKIP [detect kém]"
    print(f"    {status} | {name}")
    print(f"      MediaPipe: {mp_ok}/{NUM_FRAMES} frames detect được")
    print(f"      YOLO     : {yolo_ok}/{NUM_FRAMES} frames detect được")
    print(f"      Zero frames: {zero_frames}/{NUM_FRAMES} ({zero_ratio:.0%})")

    if zero_ratio > SKIP_THRESHOLD:
        return None

    if zero_ratio > 0:
        arr = interpolate_missing(arr)

    arr = normalize_sequence(arr)
    return arr

# ── Build dataset ─────────────────────────────────────────────────────────────

def build_dataset():
    analyze_dataset(DATASET_DIR)

    X_all, y_all = [], []
    stats        = {}

    for label_name, label_id in LABELS.items():
        folder = os.path.join(DATASET_DIR, label_name)
        if not os.path.exists(folder):
            print(f"ERROR: Không tìm thấy folder: {folder}")
            continue

        videos = (list(Path(folder).glob("*.mp4")) +
                  list(Path(folder).glob("*.avi")) +
                  list(Path(folder).glob("*.mov")))

        print(f"\n[{label_name}] {len(videos)} videos")
        print(f"{'─'*40}")
        ok, skip = 0, 0

        for i, vp in enumerate(videos):
            print(f"  [{i+1}/{len(videos)}] {vp.name}")
            kp = extract_one(str(vp), label_name)
            if kp is not None:
                X_all.append(kp)
                y_all.append(label_id)
                ok += 1
            else:
                skip += 1

        stats[label_name] = {"ok": ok, "skip": skip}
        print(f"\n  DONE [{label_name}]: OK={ok}  SKIP={skip}")
        print(f"{'─'*40}\n")

    if not X_all:
        print("ERROR: Không extract được video nào!")
        return

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    config = {
        "num_frames":     NUM_FRAMES,
        "window_sec":     WINDOW_SEC,
        "skip_threshold": SKIP_THRESHOLD,
        "input_dim":      INPUT_DIM,
        "mp_kp":          MP_KP,
        "yolo_kp":        YOLO_KP,
        "fusion":         True,
        "normalized":     True,
        "labels":         {"0": "non_fall", "1": "fall"},
        "sampling": {
            "fall":     "full video",
            "non_fall": f"last {WINDOW_SEC}s",
        },
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"{'='*50}")
    print(f"DONE")
    print(f"  X shape    : {X.shape}")
    print(f"  y shape    : {y.shape}")
    print(f"  fall       : {int((y == 1).sum())}")
    print(f"  non_fall   : {int((y == 0).sum())}")
    print(f"  num_frames : {NUM_FRAMES}")
    print(f"  input_dim  : {INPUT_DIM}")
    print(f"  normalized : True")
    print(f"  Saved to   : {OUTPUT_DIR}")
    print(f"{'='*50}")

    # Cảnh báo nếu mất balance
    fall_ok    = stats.get("fall",     {}).get("ok", 0)
    nf_ok      = stats.get("non_fall", {}).get("ok", 0)
    fall_skip  = stats.get("fall",     {}).get("skip", 0)
    nf_skip    = stats.get("non_fall", {}).get("skip", 0)
    print(f"\n  fall    : OK={fall_ok}  SKIP={fall_skip}")
    print(f"  non_fall: OK={nf_ok}  SKIP={nf_skip}")
    if abs(fall_ok - nf_ok) > 20:
        print(f"\n  WARNING: Mất balance! fall={fall_ok} vs non_fall={nf_ok}")
        print(f"  Cân nhắc tăng SKIP_THRESHOLD hoặc kiểm tra lại video bị skip")


if __name__ == "__main__":
    build_dataset()