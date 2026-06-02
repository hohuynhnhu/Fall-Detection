# Fall Detection Desktop App — Tổng quan kiến trúc

## Mục lục
1. [Cấu trúc thư mục](#1-cấu-trúc-thư-mục)
2. [Luồng khởi động](#2-luồng-khởi-động)
3. [Các module trong `src/core/`](#3-các-module-trong-srccore)
4. [Luồng xử lý chính](#4-luồng-xử-lý-chính)
5. [Pipeline nhận diện khuôn mặt](#5-pipeline-nhận-diện-khuôn-mặt)
6. [Giao diện người dùng](#6-giao-diện-người-dùng)
7. [Schema & cấu hình](#7-schema--cấu-hình)
8. [Tích hợp backend](#8-tích-hợp-backend)
9. [Training Transformer](#9-training-transformer)

---

## 1. Cấu trúc thư mục

```
Fall-Detection/
├── src/
│   ├── app.py                    # Entry point — Tkinter UI (~1200 dòng)
│   ├── schemas.py                # Pydantic models & enums
│   └── core/
│       ├── pose_engine.py        # MediaPipe + YOLO Pose fusion
│       ├── fall_detector.py      # Phân loại trạng thái & phát hiện té ngã
│       ├── camera_worker.py      # Thread chính: capture & điều phối
│       ├── person_detector.py    # YOLOv8s + ByteTrack đa người
│       ├── transformer_engine.py # Transformer sliding-window inference
│       ├── face_engine.py        # YOLOv8s + InsightFace ArcFace
│       ├── face_recognizer_if.py # Wrapper InsightFace buffalo_l (512-d)
│       ├── family_manager.py     # Đồng bộ thành viên từ backend + WebSocket
│       ├── audio_engine.py       # YAMNet phát hiện âm thanh té ngã
│       ├── appearance_tracker.py # Tracking theo màu sắc (HSV histogram)
│       └── overlay.py            # Vẽ overlay lên frame camera
├── train_tranformer/
│   ├── extract_keypoints.py      # Trích xuất sliding windows từ video
│   ├── train.py / train_k.py     # K-Fold training với early stopping
│   └── dataset/
│       ├── fall/ non_fall/       # Video training (~204 video mỗi loại)
│       ├── X.npy, y.npy          # (N, 30, 200) keypoints + labels
│       └── checkpoints1/
│           └── best_model.pth    # Trọng số mô hình đã huấn luyện
├── data/profiles/face_db.pkl     # Database embedding khuôn mặt (local)
├── clips/                        # Video clip ghi lại sự kiện té ngã
├── models/face/                  # Model weights (dlib, deprecated)
├── .env                          # Cloudinary credentials
└── run.bat                       # Launcher Windows
```

---

## 2. Luồng khởi động

**Entry point:** `src/app.py` → class `FallDetectionApp` (Tkinter)

```
1. Fetch config từ backend (GET /config)
   └── ThresholdConfig + FeatureConfig
   └── Fallback về giá trị mặc định nếu offline

2. Khởi tạo Face Recognition (_init_face_recognition)
   └── GET /family-members/all
   └── Nếu có thành viên → bật Face ID
   └── Tạo FamilyManager (đồng bộ + WebSocket)

3. Khởi động CameraWorker (threading.Thread)
   └── Luôn bật YOLO Pose + Transformer AI
   └── Bật/tắt face recognition theo cấu hình
   └── Đẩy WorkerFrame vào result_queue (maxsize=1)

4. Khởi động BackendClient (async event loop)
   └── Poll config mỗi 5 giây
   └── Gửi events fall/pose/heartbeat bất đồng bộ

5. UI polling loop mỗi 14ms (~71 FPS)
   └── Drain result_queue → _update() → render
```

---

## 3. Các module trong `src/core/`

| Module | Chức năng |
|--------|-----------|
| `pose_engine.py` | Trích xuất keypoint: MediaPipe (33 kp) + YOLO Pose (17 kp), fusion confidence-weighted → vector 200 chiều |
| `fall_detector.py` | Phân loại trạng thái (STANDING/SITTING/LYING/WALKING/FALLING), tính vận tốc, phát hiện té ngã theo rule |
| `camera_worker.py` | Thread điều phối: mở camera/RTSP, gọi các engine, quản lý tracking đa người, ghi clip |
| `person_detector.py` | YOLOv8s + ByteTrack: trả về `(track_id, bbox, confidence)` cho từng người |
| `transformer_engine.py` | Sliding window 30 frame × 200 features → Transformer → P(fall); ngưỡng 0.65 |
| `face_engine.py` | YOLOv8s detect người → InsightFace ArcFace 512-d → cosine similarity ≥ 0.35 |
| `face_recognizer_if.py` | Wrapper InsightFace buffalo_l: tải model 300 MB, trả embedding chuẩn hóa 512-d |
| `family_manager.py` | Đồng bộ thành viên từ API, lắng nghe WebSocket `/ws/desktop`, thread-safe |
| `audio_engine.py` | YAMNet (TensorFlow-Hub): trigger sau té ngã, ghi 3s audio, phân loại 521 lớp AudioSet |
| `appearance_tracker.py` | Tracking ổn định bằng HSV histogram + shape (shoulder width, body height) |
| `overlay.py` | Badge trạng thái, velocity bar, skeleton màu theo state, banner cảnh báo té ngã |

---

## 4. Luồng xử lý chính

```
Frame camera (30 fps)
        │
        ▼
┌─────────────────────────────────────┐
│  POSE EXTRACTION (pose_engine.py)   │
│  MediaPipe 33 kp + YOLO Pose 17 kp  │
│  → Fusion → BodyMetrics + kp(200,)  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  STATE CLASSIFICATION               │
│  LYING:    body_angle ≥ 75°         │
│            OR aspect_ratio ≤ 0.45   │
│  SITTING:  torso_ratio < 0.42       │
│  WALKING:  vel_x > 20 px/s          │
│            + knee lift xen kẽ       │
│  STANDING: mặc định                 │
└──────────────────┬──────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐   ┌──────────────────────┐
│  RULE-BASED   │   │  TRANSFORMER AI      │
│  FALL DETECT  │   │  30 frame × 200 feat │
│               │   │  → P(fall) ≥ 0.65    │
│  vel_y > 80   │   │  Infer mỗi 8 frame   │
│  LYING ≥ 5f   │   └──────────────────────┘
│  trans < 2s   │
│  hip_drop >   │
│  shld_drop    │
└───────┬───────┘
        │
        ▼
┌─────────────────────────────────────┐
│  QUYẾT ĐỊNH CUỐI (app.py)           │
│  ai_fall = AI AND vel_y > 150       │
│           AND was_upright           │
│           AND lying_duration < 1s   │
│  final = ai_fall AND rule_fall      │
└──────────────────┬──────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐   ┌──────────────────────┐
│  UI ALERT     │   │  CLIP RECORDING      │
│  Beep × 3     │   │  Pre: 300 frame      │
│  Banner đỏ    │   │  Post: 150 frame     │
│  Event log    │   │  Upload Cloudinary   │
└───────────────┘   └──────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  POST /events/fall (async)          │
└─────────────────────────────────────┘
```

**Logic quyết định cuối (app.py):**
```python
ai_sees_fall = (
    tr.ready and tr.is_fall
    and vel_y > 150
    and current_lying
    and was_upright
    and lying_duration < 1.0
)
final_fall = ai_sees_fall and rule_fall  # cả hai phải đồng ý
```

---

## 5. Pipeline nhận diện khuôn mặt

```
Frame camera
    │
    ▼
Person Detection (YOLOv8s + ByteTrack)
    │
    ├── track_id MỚI → Crop bbox → InsightFace ArcFace → 512-d embedding
    │                                                         │
    │                                                         ▼
    │                                              So sánh face_db.pkl
    │                                              cosine_sim ≥ 0.35 → MATCH
    │
    └── track_id ĐÃ BIẾT → Appearance matching (HSV histogram)
                            → Bỏ qua face recognition frame này
```

**Cấu trúc database (`data/profiles/face_db.pkl`):**
```python
{
  "person_id_abc": {
    "name": "Nguyễn Văn A",
    "role": "family",       # hoặc "caregiver"
    "is_patient": True,     # theo dõi tư thế bệnh nhân
    "sample_count": 3,
    "encodings": [ndarray(512), ...],
    "added_at": 1234567890.0
  }
}
```

**Đồng bộ thành viên (FamilyManager):**
1. Fetch `GET /family-members/all` → download ảnh → trích embedding → lưu local
2. WebSocket `/ws/desktop` nhận real-time: `new_member`, `remove_member`
3. `FaceEngine` và `FamilyManager` dùng chung 1 instance `FaceRecognizer` (tránh load model 2 lần)

**Đăng ký khuôn mặt mới (UI):**
1. Nhập tên + vai trò (family/caregiver)
2. Chọn có phải bệnh nhân cần theo dõi không
3. Chụp frame hiện tại từ camera
4. Trích ArcFace 512-d → lưu vào `face_db.pkl`
5. Sync lên backend: `POST /family-members/register`

---

## 6. Giao diện người dùng

**Cửa sổ chính:** Tkinter, 1440×860 px, theme dark (tiếng Việt)

```
┌──────────────────────────────────────────────────────────────┐
│ HEADER: FALL DETECTION SYSTEM        ● ONLINE     14:32:15   │
├─────────────────────┬────────────────────────────────────────┤
│                     │ 1. ĐIỀU KHIỂN                          │
│                     │    [▶ BẮT ĐẦU] [■ DỪNG] [↺ Reset]    │
│   CAMERA FEED       │    [● YOLO] [● AI] [● FACE]           │
│   (scalable)        │    API: [http://localhost:8000 ____]   │
│                     │                                        │
│   Overlay:          │ 2. TRẠNG THÁI                          │
│   - Badge state     │    STANDING          AI: SAFE 95%      │
│   - Velocity bar    │    Confidence: 87%   Buffer: 25/30     │
│   - Skeleton màu    │                                        │
│   - Bounding boxes  │ 3. CẢNH BÁO TÉ NGÃ                    │
│   - Fall banner     │    ⚠ TÉ NGÃ !  [nhấp nháy]           │
│                     │    Nguồn: Rules + AI  Số lần: 2       │
│                     │                                        │
│                     │ 4. NHẬN DIỆN KHUÔN MẶT                │
│                     │    Nguyễn Văn A  [API: BẬT]           │
│                     │    Confidence: 92%                     │
│                     │                                        │
│                     │ 5. BỆNH NHÂN THEO DÕI                  │
│                     │    Bà Mẹ     STANDING   14:32:15       │
│                     │    Anh Trai  WALKING    14:31:42       │
│                     │                                        │
│                     │ 6. CHỈ SỐ                              │
│                     │    Vel Y: 45  Góc: 12.3°  FPS: 28     │
│                     │                                        │
│                     │ 7. ÂM THANH / SỰ KIỆN LOG             │
└─────────────────────┴────────────────────────────────────────┘
```

**Màu sắc theo trạng thái:**

| Trạng thái | Màu |
|-----------|-----|
| STANDING  | `#3bff8a` (xanh lá) |
| SITTING   | `#4a9eff` (xanh dương) |
| WALKING   | `#b04aff` (tím) |
| LYING     | `#ffaa3b` (cam) |
| FALLING   | `#ff3b3b` (đỏ) |
| UNKNOWN   | `#888888` (xám) |

---

## 7. Schema & cấu hình

**`schemas.py` — các model chính:**

```python
class ThresholdConfig(BaseModel):
    # Phân loại tư thế
    body_angle_lying: float = 75.0       # ≥ 75° → LYING
    aspect_ratio_lying: float = 0.45     # ≤ 0.45 → LYING
    torso_ratio_sitting: float = 0.42    # < 0.42 → SITTING

    # Phát hiện té ngã
    fall_velocity_threshold: float = 80.0    # px/s
    fall_confirm_frames: int = 10            # frame LYING cần có
    fall_transition_max_s: float = 2.0       # đứng → nằm tối đa 2s
    sleep_confirm_frames: int = 150          # nằm ≥ 150 frame = ngủ

    # Đi bộ
    walk_velocity_threshold: float = 20.0
    walk_knee_lift_threshold: float = 0.08

class FeatureConfig(BaseModel):
    enable_face_recognition: bool = True
    enable_patient_pose_notification: bool = True
    enable_sound_detection: bool = True
    sleep_as_fall: bool = False
    sound_listen_seconds: float = 3.0
```

**Các event gửi lên backend:**

| Event | Endpoint | Dữ liệu chính |
|-------|----------|----------------|
| Té ngã | `POST /events/fall` | velocity, angle, confidence, clip_url |
| Thay đổi tư thế | `POST /events/pose` | state, prev_state, velocity |
| Phát hiện người | `POST /events/person-detected` | person_count, confidence |
| Tư thế bệnh nhân | `POST /events/patient-pose` | person_id, state, prev_state |
| Heartbeat | `POST /events/heartbeat` | fps, current_state |

---

## 8. Tích hợp backend

**Desktop là HTTP client kết nối FastAPI backend tại `http://localhost:8000`** (cấu hình được trong UI).

```
Desktop App                        FastAPI Backend
    │                                    │
    │── GET /health ───────────────────> │  kiểm tra kết nối
    │── GET /config ───────────────────> │  fetch ThresholdConfig + FeatureConfig
    │── POST /events/* ───────────────> │  gửi sự kiện (async queue)
    │── GET /family-members/all ──────> │  fetch thành viên
    │── POST /family-members ─────────> │  đăng ký thành viên mới
    │                                    │
    │<── WebSocket /ws/desktop ──────── │  push: new_member, remove_member
    │<── (poll /config mỗi 5s) ──────── │  cập nhật ngưỡng/features real-time
```

**Hoạt động offline:** nếu backend không kết nối được, app dùng giá trị mặc định từ `ThresholdConfig` và `FeatureConfig`.

---

## 9. Training Transformer

**Pipeline huấn luyện (`train_tranformer/`):**

**Bước 1 — Trích xuất keypoint (`extract_keypoints.py`):**
- Input: video fall/non_fall (~3–6s mỗi video)
- Tiền xử lý: CLAHE nếu ảnh tối (mean brightness < 80)
- Trích xuất: MediaPipe 132 feat + YOLO Pose 68 feat = **200 feat/frame**
- Sliding window: 30 frame, nội suy frame thiếu
- Chuẩn hóa per-sequence (μ=0, σ=1) — bắt buộc khớp inference
- Output: `X.npy` (N, 30, 200), `y.npy` (N,)

**Bước 2 — Huấn luyện (`train.py`):**

```
Kiến trúc PoseTransformer:
  Linear(200 → 128) + LayerNorm + ReLU + Dropout
  + Learnable CLS token + Positional embeddings (31, 128)
  → TransformerEncoder: 2 layers, 4 heads, d_model=128
  → LayerNorm → Linear(128 → 32) → GELU → Linear(32 → 2)
```

| Hyperparameter | Giá trị |
|----------------|---------|
| Batch size | 16 |
| Learning rate | 5e-4 |
| Weight decay | 5e-3 |
| Dropout | 0.4 |
| Label smoothing | 0.1 |
| Early stopping | patience=15 |
| Optimizer | AdamW + CosineAnnealing |

**Data augmentation:** horizontal flip, speed variation (0.8–1.2×), Gaussian noise (σ=0.01), frame dropout (10%)

**Kết quả 5-Fold CV:**
- Accuracy: **97.8% ± 1.2%**
- Miss rate (false negative): **1.5%**

**Bước 3 — Inference thời gian thực:**
- Load `checkpoints1/best_model.pth`
- Buffer 30 frame, infer mỗi 8 frame
- Ngưỡng: P(fall) ≥ 0.65 → cảnh báo
