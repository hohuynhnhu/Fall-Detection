# Fall Detection Desktop System

Hệ thống phát hiện té ngã realtime chạy trên máy tính, kết hợp **Rule-based** và **AI Transformer**.

---

## Cấu trúc thư mục

```
src/
├── core/
│   ├── pose_engine.py        # Trích xuất keypoints (MediaPipe + YOLO)
│   ├── fall_detector.py      # Phát hiện té ngã rule-based
│   ├── transformer_engine.py # Phân loại fall/non-fall bằng AI
│   ├── camera_worker.py      # Thread điều phối camera + các engine
│   ├── face_engine.py        # Nhận diện khuôn mặt (dlib)
│   ├── audio_engine.py       # Phát hiện âm thanh (YAMNet)
│   └── overlay.py            # Vẽ UI lên camera frame
├── services/
│   └── backend_client.py     # Giao tiếp với backend API
├── app.py                    # Entry point — Tkinter UI
├── schemas.py                # Pydantic data models
└── yolov8n-pose.pt           # YOLO Pose model
```

---

## Pipeline

```
Camera / RTSP
      ↓
pose_engine        →  BodyMetrics + raw_kp (200,)
      ├── fall_detector      →  is_falling  (Rule-based)
      ├── transformer_engine →  is_fall     (AI Transformer)
      ├── face_engine        →  tên người   (optional)
      └── audio_engine       →  âm thanh    (optional)
      ↓
camera_worker  →  WorkerFrame  →  queue
      ↓
app.py  →  kết hợp kết quả  →  hiển thị + gửi backend
```

---

## Cách phát hiện té ngã

### Rule-based (`fall_detector.py`)

| Điều kiện              | Mô tả              |
| ---------------------- | ------------------ |
| `velocity_y > 80 px/s` | Hạ xuống nhanh     |
| `was STANDING/WALKING` | Trước đó đứng/đi   |
| `LYING >= 5 frames`    | Đang nằm liên tiếp |
| `transition < 2s`      | Chuyển đổi nhanh   |

### AI Transformer (`transformer_engine.py`)

- Input: sliding window **30 frames × 200 features** (MediaPipe 33 kp + YOLO 17 kp)
- Normalize per-sequence trước khi infer
- Output: `P(fall)` — ngưỡng mặc định **0.6**
- Kết quả train: **Mean 97.8% ± 1.2%**, Miss Rate **1.5%**

### Kết hợp

```
any_fall = rule_fall AND ai_fall
```

---

## Các tính năng

| Tính năng      | Mô tả                      | Bật/Tắt           |
| -------------- | -------------------------- | ----------------- |
| YOLO Pose      | Hỗ trợ MediaPipe khi tối   | Checkbox UI       |
| Face ID        | Nhận diện khuôn mặt (dlib) | Checkbox UI / API |
| AI Transformer | Phân loại fall/non-fall    | Checkbox UI       |
| Audio (YAMNet) | Phát hiện âm thanh té ngã  | API backend       |
| Sleep-as-fall  | Nằm lâu = té ngã           | API backend       |
| RTSP           | Kết nối camera IP          | Nhập URL          |

---

## Cài đặt

```bash
pip install mediapipe ultralytics torch opencv-python pillow pydantic dlib
```

**Model files cần có:**

```
models/face/shape_predictor_68_face_landmarks.dat
models/face/dlib_face_recognition_resnet_model_v1.dat
train_tranformer/dataset_1/checkpoints1/best_model.pth
```

---

## Chạy

```bash
cd src
python app.py
```

---

## Dataset & Training

```
fall:     204 video (~3s) — đi → té → nằm
non_fall: 204 video (~3-6s) — đứng/ngồi → nằm

Sampling:
  fall     → toàn bộ video
  non_fall → 3s cuối (lúc hành động nằm xuống)

num_frames = 30 | input_dim = 200 | K-Fold = 5
```

---

## Ngưỡng mặc định

| Tham số                   | Giá trị | Mô tả                     |
| ------------------------- | ------- | ------------------------- |
| `fall_velocity_threshold` | 80 px/s | Tốc độ hạ để trigger      |
| `body_angle_lying`        | 65°     | Góc thân để coi là nằm    |
| `aspect_ratio_lying`      | 0.55    | H/W bbox để coi là nằm    |
| `fall_confirm_frames`     | 5       | Frame LYING liên tiếp     |
| `fall_transition_max_s`   | 2.0s    | Thời gian tối đa đứng→nằm |
| `fall_threshold` (AI)     | 0.6     | P(fall) để kết luận       |
