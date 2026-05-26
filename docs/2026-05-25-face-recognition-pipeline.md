# 2026-05-25 — Face Recognition Pipeline: Thiết kế & Bug

## Tổng quan

Ngày hôm nay thiết kế lại toàn bộ pipeline nhận diện khuôn mặt từ **local-only** sang **mobile → backend → desktop**, đồng thời phát hiện và fix một số bug nghiêm trọng.

---

## Pipeline cũ (trước hôm nay)

```
Desktop UI → dlib encode → lưu face_db.pkl (local)
Desktop UI → POST /family-members (chỉ metadata: tên, role, is_patient)
Camera     → FaceEngine so sánh với face_db.pkl
```

**Hạn chế:** Face encoding chỉ tồn tại trên máy desktop. Xóa `face_db.pkl` là mất hết. Không thể đăng ký từ mobile.

---

## Pipeline mới (sau hôm nay)

```
Mobile App
  └─ POST /family-members/register { name, role, is_patient, face_image_base64 }
          │
          ▼
Backend (FastAPI)
  ├─ Lưu ảnh vào storage → trả face_image_url
  └─ WebSocket push /ws/desktop
       { type: "new_member", person_id, name, is_patient, face_image_url }
          │
          ▼
Desktop App — khi bấm BẮT ĐẦU
  ├─ GET /family-members/all → danh sách { name, is_patient, face_image_url }
  ├─ Download từng ảnh từ URL
  ├─ dlib extract → 128-d encoding → lưu vào FamilyManager (in-memory dict)
  └─ WebSocket lắng nghe member mới / xóa real-time
          │
          ▼
Camera chạy — mỗi 5 frames
  ├─ dlib detect khuôn mặt
  ├─ extract encoding 128-d
  ├─ FamilyManager.match_person() → so Euclidean distance
  └─ Nếu is_patient → gửi PatientPoseEvent về backend/mobile
```

---

## Files tạo mới

### `src/core/family_manager.py`
Class `FamilyManager` — quản lý toàn bộ face data từ backend:

| Method | Chức năng |
|--------|-----------|
| `start(detector, predictor, recognizer)` | Inject dlib models từ FaceEngine, load API, bắt WS thread |
| `stop()` | Dừng WS listener |
| `match_person(encoding)` | So khớp → `{person_id, name, is_patient, confidence}` hoặc `None` |
| `add_encoding(...)` | Thêm encoding khi đăng ký qua UI desktop |
| `remove(person_id)` | Xóa khỏi in-memory dict |
| `_load_from_api()` | GET /family-members/all → download → extract |
| `_download_and_encode()` | Download ảnh từ URL → dlib extract → lưu dict |
| `_ws_loop()` | Async WS loop với auto-reconnect 5s |
| `_handle_ws_message()` | Xử lý `new_member` / `remove_member` |

**Thread-safe:** dùng `threading.Lock` cho `_members` dict.  
**Log callback:** `on_log(msg)` để hiện log lên UI panel.

---

## Files sửa đổi

### `src/schemas.py`
Thêm:
- `FeatureConfig.enable_patient_pose_notification: bool = True` — bật/tắt gửi thông báo tư thế bệnh nhân từ mobile
- `FamilyMemberWithImage` — schema có `face_image_url` cho `/family-members/all`
- `FamilyMembersAllResponse`
- `NewMemberWSMessage` — WebSocket push message từ backend

### `src/core/face_engine.py`
- Thêm `RecognizedPerson.is_patient: bool = False` — is_patient đi kèm kết quả nhận diện
- Thêm `FaceEngine.set_family_manager(fm)` — inject FamilyManager
- Sửa `_match()`:
  - Nếu có FamilyManager → dùng `family_manager.match_person()` (network-backed)
  - Fallback → local `face_db.pkl`
  - Set `is_patient` trực tiếp trên `RecognizedPerson`

### `src/core/camera_worker.py`
- Thêm param `family_manager: Optional[Any]`
- Sau khi `FaceEngine()` init xong → `family_manager.start(fe.detector, fe.predictor, fe.recognizer)`
- Log chi tiết: `use_face`, `dlib_ok`, `feature_on`, trạng thái init

### `src/app.py`
- Thêm `_init_face_recognition()` — tự động bật/tắt face recognition dựa vào API:
  - GET /family-members/all
  - Có `face_image_url` → `use_face=True`, tạo `FamilyManager`
  - Không có ảnh → `use_face=False`, không load dlib
- Xóa `_pending_enrollments`, `_on_new_member`, `_handle_new_member`, `_do_enroll` (thay bằng FamilyManager)
- `_update_patient_monitoring()` dùng `person.is_patient` thay vì `fe.is_patient()` (local DB)
- Panel "BỆNH NHÂN THEO DÕI" thêm badge `[API: BẬT/TẮT]`
- `FamilyManagementWindow._add()` đồng bộ encoding vào FamilyManager khi đăng ký qua UI

### `src/services/backend_client.py`
- Xóa WS listener (FamilyManager tự quản lý WS)
- Fix `stop()`: thay `loop.stop()` bằng enqueue sentinel `None` → tránh `RuntimeError`
- `_event_send_loop` thoát khi nhận sentinel `None`

---

## API Backend cần implement

```
GET  /family-members/all
Response: { "members": [{ person_id, name, role, is_patient, face_image_url }] }

POST /family-members/register
Body: { name, role, is_patient, face_image_base64 }
Response: { person_id, name, role, is_patient }

WS   /ws/desktop  ← Desktop kết nối lắng nghe
Push new_member:    { type, person_id, name, is_patient, face_image_url }
Push remove_member: { type, person_id }

PATCH /config/features
Body (partial): {
  enable_face_recognition: bool,
  enable_patient_pose_notification: bool,
  enable_sound_detection: bool,
  sleep_as_fall: bool
}
```

---

## Bug tìm thấy & đã fix

### Bug 1 — `is_patient` luôn False khi dùng FamilyManager ⚠️ NGHIÊM TRỌNG

**Nguyên nhân:**
```python
# app.py — _update_patient_monitoring()
if not fe.is_patient(person.person_id):   # ← chỉ check local pickle DB
    continue
```
Người đăng ký qua mobile → FamilyManager → **không có trong pickle DB** → `is_patient` luôn `False` → không bao giờ gửi `PatientPoseEvent`.

**Fix:**
```python
# Thêm is_patient vào RecognizedPerson dataclass
@dataclass
class RecognizedPerson:
    ...
    is_patient: bool = False   # set trong _match() từ FamilyManager

# app.py — dùng trực tiếp
if not person.is_patient:
    continue
```

---

### Bug 2 — `RuntimeError: Event loop stopped before Future completed`

**Nguyên nhân:** `BackendClient.stop()` gọi `loop.stop()` trong khi `run_until_complete(_main())` đang chạy → loop dừng đột ngột → exception.

**Fix:**
```python
def stop(self):
    self._running = False
    # Enqueue sentinel để unblock _event_send_loop
    if self._loop and self._queue:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        except Exception:
            pass
```
`_event_send_loop` nhận `None` → `break` → `gather()` hoàn thành → `run_until_complete` thoát sạch.

---

### Bug 3 — Face recognition không hoạt động (chưa fix hoàn toàn) 🔴

**Triệu chứng:** `[Face] use_face=False` dù dlib có sẵn.

**Nguyên nhân:** `_init_face_recognition()` trả `False` — một trong các lý do:
- Backend offline khi bấm BẮT ĐẦU
- `GET /family-members/all` trả HTTP ≠ 200
- Response không có field `face_image_url`
- Response `members` rỗng

**Cách debug:** Xem console sau khi bấm BẮT ĐẦU:
```
[Face] GET http://localhost:8000/family-members/all
[Face] HTTP ???
[Face] Nhận ? thành viên từ API
[Face]   Tên — ✓/✗ có ảnh | keys=[...]
```

**Hướng fix:** Tùy vào output console — backend cần implement đúng `/family-members/all` với field `face_image_url`.

---

## Điều kiện để face recognition hoạt động

| Điều kiện | Kiểm tra |
|-----------|----------|
| Backend online | URL đúng, `/health` trả 200 |
| `/family-members/all` trả HTTP 200 | Backend implement endpoint |
| Response có field `face_image_url` | Không phải `face_image` hay `image_url` |
| URL ảnh download được | Không bị CORS, auth |
| Ảnh chứa khuôn mặt rõ | dlib detect được |
| dlib model files có đủ | `models/face/*.dat` |
