<div align="center">

# 🏥 Smart Fall Detection

**Hệ thống phát hiện té ngã thông minh cho người cao tuổi**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![YOLO](https://img.shields.io/badge/YOLOv8-Pose-FF6B35?logo=ultralytics&logoColor=white)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

<img src="docs/architecture.png" alt="System Architecture" width="700"/>

> Kết hợp **YOLO Pose + GCN + TCN** để phát hiện té ngã theo thời gian thực,  
> tích hợp nhận diện khuôn mặt và **cá nhân hóa theo bệnh án** từng người cao tuổi.

</div>

---

## 📌 Tổng quan

**Smart Fall Detection** là hệ thống giám sát thông minh giúp phát hiện té ngã của người cao tuổi trong môi trường gia đình. Hệ thống không chỉ phát hiện té ngã mà còn **nhận diện từng thành viên** trong gia đình để tự động điều chỉnh mức độ nhạy cảnh báo dựa trên **hồ sơ bệnh án cá nhân**.

---

## ✨ Tính năng chính

- 🦴 **Phát hiện dáng người** — YOLO Pose trích xuất 17 keypoints skeleton theo thời gian thực
- 🧠 **Nhận diện té ngã** — GCN + TCN học đặc trưng không gian & thời gian của chuyển động
- 👤 **Nhận diện thành viên** — dlib xác định danh tính khuôn mặt trong gia đình
- 📋 **Cá nhân hóa theo bệnh án** — tự động tăng độ nhạy nếu người dùng có tiền sử té ngã, Parkinson, xương khớp yếu...
- 🚨 **Cảnh báo tức thời** — gửi thông báo ngay khi phát hiện té ngã
- 🐳 **Docker ready** — triển khai dễ dàng với Docker Compose

---

## 🏗️ Kiến trúc hệ thống

```
Camera / Video Input
        ↓
┌─────────────────────────────┐
│       Detector Service       │
│                             │
│  YOLO Pose → Keypoints      │
│  dlib      → Face ID        │
│  GCN + TCN → Fall Detection │
└────────────┬────────────────┘
             │ Redis Queue
┌────────────▼────────────────┐
│         API Service          │
│                             │
│  FastAPI  → REST API        │
│  Medical  → Điều chỉnh ngưỡng│
│  Alert    → Gửi cảnh báo    │
└────────────┬────────────────┘
             │
       PostgreSQL DB
```

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Pose Estimation | YOLOv8 Pose |
| Fall Detection | GCN + TCN |
| Face Recognition | dlib |
| Backend API | FastAPI |
| Database | PostgreSQL |
| Message Queue | Redis |
| Containerization | Docker + Docker Compose |

---

## 📁 Cấu trúc dự án

```
smart-fall-detection/
├── data/
│   ├── raw/                   # Dữ liệu thô
│   ├── processed/             # Dữ liệu đã xử lý
│   └── profiles/              # Hồ sơ thành viên gia đình
├── models/
│   ├── yolo/                  # YOLOv8 Pose weights
│   ├── gcn_tcn/               # GCN + TCN checkpoints
│   └── face/                  # dlib face models
├── src/
│   ├── detection/
│   │   ├── pose/              # YOLO Pose pipeline
│   │   ├── fall/              # GCN + TCN inference
│   │   └── face/              # dlib face recognition
│   ├── medical/               # Xử lý bệnh án, điều chỉnh ngưỡng
│   ├── alert/                 # Gửi cảnh báo
│   └── api/                   # FastAPI routes & models
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.detector
├── docker-compose.yml
└── README.md
```

---

## 🚀 Cài đặt & Chạy

### Yêu cầu

- Docker & Docker Compose
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) *(nếu dùng GPU)*

### 1. Clone repo

```bash
git clone https://github.com/your-username/smart-fall-detection.git
cd smart-fall-detection
```

### 2. Cấu hình môi trường

```bash
cp .env.example .env
# Chỉnh sửa .env theo cấu hình của bạn
```

### 3. Chạy với Docker

```bash
# Build và khởi động toàn bộ hệ thống
docker-compose up --build

# Chạy nền
docker-compose up -d --build
```

### 4. Truy cập API

```
http://localhost:8000/docs
```

---

## ⚙️ Cấu hình `.env`

```env
# Database
DATABASE_URL=postgresql://admin:yourpassword@db:5432/fall_detection

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Fall Detection
FALL_SENSITIVITY_DEFAULT=0.7
FALL_SENSITIVITY_HIGH=0.5     # Dùng cho người có bệnh án nguy cơ cao
```

---

## 🔬 Nghiên cứu & Mô hình

### Pipeline phát hiện té ngã

1. **YOLO Pose** trích xuất 17 keypoints từ khung hình
2. Keypoints được đưa vào **GCN** để học mối quan hệ không gian giữa các khớp
3. **TCN** học chuỗi chuyển động theo thời gian
4. Kết hợp đặc trưng → phân loại: `normal` / `falling` / `fallen`

### Cá nhân hóa theo bệnh án

Khi hệ thống nhận diện được khuôn mặt thành viên gia đình, API sẽ tra cứu hồ sơ bệnh án và điều chỉnh ngưỡng phát hiện:

| Tình trạng | Độ nhạy |
|---|---|
| Bình thường | 0.7 |
| Tiền sử té ngã | 0.5 |
| Parkinson / xương khớp yếu | 0.4 |

---

## 📄 License

MIT License — xem [LICENSE](LICENSE) để biết thêm chi tiết.

---

<div align="center">
Made with ❤️ for elderly care
</div>
