"""
src/services/backend_client.py — HTTP client gửi events lên backend

── Desktop app GỌI (client → backend) ────────────────────────────────────────
  GET    /health                    — kiểm tra kết nối lúc khởi động
  GET    /config?device_id=cam_0    — lấy ThresholdConfig + FeatureConfig (mỗi 30s)
  POST   /events/fall               — báo cáo sự kiện té ngã
  POST   /events/pose               — báo cáo thay đổi tư thế (mỗi 30 frames)
  POST   /events/person-detected    — báo cáo nhận diện khuôn mặt (debounce 5s)
  POST   /device/status             — heartbeat trạng thái thiết bị (mỗi 5s)
  GET    /family-members            — lấy danh sách thành viên
  POST   /family-members            — thêm thành viên mới
  DELETE /family-members/{id}       — xóa thành viên

── Mobile app GỌI (mobile → backend) — backend cần implement ─────────────────
  PATCH  /config/features           — bật/tắt tính năng từ mobile app
    Request body:
      { "enable_face_recognition": bool,
        "enable_sound_detection":  bool,
        "sleep_as_fall":           bool,
        "sound_listen_seconds":    float }   ← tất cả đều optional (partial update)
    Response: { "ok": true, "features": { ...updated FeatureConfig... } }

  PATCH  /config/thresholds         — chỉnh ngưỡng phát hiện từ mobile app
    Request body: bất kỳ field nào của ThresholdConfig (partial update)
    Response: { "ok": true, "thresholds": { ...updated ThresholdConfig... } }

  Desktop tự đọc lại qua GET /config mỗi 30s và áp dụng ngay khi có thay đổi.
"""
from __future__ import annotations
import asyncio
import threading
import time
import httpx
from typing import Optional, Callable, List, Union

from schemas import (
    ThresholdConfig, FeatureConfig, ConfigResponse,
    FallEvent, PoseEvent, PersonDetectedPayload,
    DeviceStatusPayload, DeviceStatusEnum,
    AddFamilyMemberPayload, FamilyMember, FamilyMembersResponse,
    PoseState,
)

_AnyEvent = Union[FallEvent, PoseEvent, PersonDetectedPayload]


class BackendClient:
    def __init__(
        self,
        base_url:          str  = "http://localhost:8000",
        camera_id:         str  = "cam_0",
        on_config_update:  Optional[Callable[[ThresholdConfig], None]] = None,
        on_feature_update: Optional[Callable[[FeatureConfig], None]]   = None,
        status_interval:   float = 5.0,
        config_interval:   float = 30.0,
    ):
        self.base_url          = base_url.rstrip("/")
        self.camera_id         = camera_id
        self.on_config_update  = on_config_update
        self.on_feature_update = on_feature_update
        self.status_interval   = status_interval
        self.config_interval   = config_interval

        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread]          = None
        self._queue:  Optional[asyncio.Queue]             = None
        self._running = False

        self.current_fps:      float         = 0.0
        self.current_pose:     PoseState      = PoseState.UNKNOWN
        self.current_features: FeatureConfig  = FeatureConfig()
        self._start_time:      float          = 0.0

        self.connected:  bool          = False
        self.last_error: Optional[str] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        self._running    = True
        self._start_time = time.time()
        self._thread     = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self):
        self._queue = asyncio.Queue(maxsize=200)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=3.0) as client:
            await self._check_health(client)
            await asyncio.gather(
                self._event_send_loop(client),
                self._config_poll_loop(client),
                self._device_status_loop(client),
            )

    # ── Public send API ────────────────────────────────────────────────────────

    def send_fall(self, event: FallEvent):
        self._enqueue(event)

    def send_pose(self, event: PoseEvent):
        self._enqueue(event)

    def send_person_detected(self, event: PersonDetectedPayload):
        self._enqueue(event)

    def update_stats(self, fps: float, pose: PoseState):
        self.current_fps  = fps
        self.current_pose = pose

    # ── Blocking helpers (UI startup / management dialogs) ────────────────────

    def fetch_config_sync(self) -> Optional[ThresholdConfig]:
        try:
            r = httpx.get(
                f"{self.base_url}/config",
                params={"device_id": self.camera_id},
                timeout=3.0,
            )
            if r.status_code == 200:
                resp = ConfigResponse(**r.json())
                self.current_features = resp.features
                return resp.thresholds
        except Exception as e:
            self.last_error = str(e)
        return None

    def fetch_features_sync(self) -> Optional[FeatureConfig]:
        """Force-fetch FeatureConfig ngay (không đợi polling 30s)."""
        try:
            r = httpx.get(
                f"{self.base_url}/config",
                params={"device_id": self.camera_id},
                timeout=3.0,
            )
            if r.status_code == 200:
                feat = ConfigResponse(**r.json()).features
                self.current_features = feat
                return feat
        except Exception as e:
            self.last_error = str(e)
        return None

    def fetch_family_members_sync(self) -> List[FamilyMember]:
        try:
            r = httpx.get(f"{self.base_url}/family-members", timeout=3.0)
            if r.status_code == 200:
                return FamilyMembersResponse(**r.json()).members
        except Exception as e:
            self.last_error = str(e)
        return []

    def add_family_member_sync(self, payload: AddFamilyMemberPayload) -> bool:
        try:
            r = httpx.post(
                f"{self.base_url}/family-members",
                json=payload.dict(),
                timeout=3.0,
            )
            return r.status_code in (200, 201)
        except Exception as e:
            self.last_error = str(e)
            return False

    def remove_family_member_sync(self, person_id: str) -> bool:
        try:
            r = httpx.delete(
                f"{self.base_url}/family-members/{person_id}",
                timeout=3.0,
            )
            return r.status_code in (200, 204)
        except Exception as e:
            self.last_error = str(e)
            return False

    # ── Internal ───────────────────────────────────────────────────────────────

    def _enqueue(self, item: _AnyEvent):
        if self._loop and self._queue:
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
            except asyncio.QueueFull:
                pass  # drop nếu queue tràn

    async def _check_health(self, client: httpx.AsyncClient):
        try:
            r = await client.get("/health")
            self.connected = r.status_code == 200
        except Exception as e:
            self.connected  = False
            self.last_error = str(e)

    async def _event_send_loop(self, client: httpx.AsyncClient):
        """Xử lý mọi loại event trong queue (fall, pose, person-detected)."""
        _ROUTES = {
            FallEvent:             "/events/fall",
            PoseEvent:             "/events/pose",
            PersonDetectedPayload: "/events/person-detected",
        }
        while self._running:
            try:
                event: _AnyEvent = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                route = _ROUTES.get(type(event))
                if route:
                    r = await client.post(route, json=event.dict())
                    self.connected = r.status_code < 500
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.connected  = False
                self.last_error = str(e)
                await asyncio.sleep(0.5)

    async def _config_poll_loop(self, client: httpx.AsyncClient):
        while self._running:
            await asyncio.sleep(self.config_interval)
            try:
                r = await client.get(
                    "/config", params={"device_id": self.camera_id}
                )
                if r.status_code == 200:
                    resp = ConfigResponse(**r.json())
                    self.connected = True
                    if self.on_config_update:
                        self.on_config_update(resp.thresholds)
                    if self.on_feature_update:
                        # Only fire callback when features actually changed
                        if resp.features != self.current_features:
                            self.current_features = resp.features
                            self.on_feature_update(resp.features)
            except Exception as e:
                self.connected  = False
                self.last_error = str(e)

    async def _device_status_loop(self, client: httpx.AsyncClient):
        while self._running:
            await asyncio.sleep(self.status_interval)
            payload = DeviceStatusPayload(
                device_id      = self.camera_id,
                timestamp      = time.time(),
                status         = DeviceStatusEnum.ONLINE,
                fps            = self.current_fps,
                current_pose   = self.current_pose,
                uptime_seconds = time.time() - self._start_time,
            )
            try:
                await client.post("/device/status", json=payload.dict())
            except Exception:
                pass
