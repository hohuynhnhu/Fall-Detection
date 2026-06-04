"""
src/services/backend_client.py — HTTP client gửi events lên backend
"""
from __future__ import annotations
import asyncio
import threading
import time
import httpx
from typing import Optional, Callable, List, Union

from schemas import (
    ThresholdConfig, FeatureConfig,
    FallEvent, PoseEvent, PersonDetectedPayload, PatientPoseEvent,
    HeartbeatPayload, FaceLogPayload,
    AddFamilyMemberPayload, FamilyMember, FamilyMembersResponse,
    PoseState,
)

_AnyEvent = Union[FallEvent, PoseEvent, PersonDetectedPayload, PatientPoseEvent, FaceLogPayload]

_JSON_HEADERS = {"Content-Type": "application/json"}

# Timeout (giây) theo loại event — quan trọng thì dài hơn
_TIMEOUT: dict = {
    FallEvent:             10.0,
    PatientPoseEvent:      10.0,
    PoseEvent:              3.0,
    PersonDetectedPayload:  2.0,
    FaceLogPayload:         2.0,
}

# Số lần retry theo loại event — không quan trọng thì không retry
_MAX_ATTEMPTS: dict = {
    FallEvent:             3,
    PatientPoseEvent:      3,
    PoseEvent:             1,
    PersonDetectedPayload: 1,  # drop ngay nếu fail, không block queue
    FaceLogPayload:        1,
}


def _serialize(event: _AnyEvent) -> bytes:
    """Serialize Pydantic model → JSON bytes, handle enum values correctly."""
    return event.json().encode("utf-8")


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
        self.config_interval   = 5.0

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
        if self._loop and self._queue:
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            except Exception:
                pass

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception:
            pass
        finally:
            self._loop.close()

    async def _main(self):
        self._queue = asyncio.Queue(maxsize=200)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=10.0) as client:
            await self._check_health(client)
            await asyncio.gather(
                self._event_send_loop(client),
                self._config_poll_loop(client),
                self._heartbeat_loop(client),
            )

    # ── Public send API ────────────────────────────────────────────────────────

    def send_fall(self, event: FallEvent):
        self._enqueue(event)

    def send_pose(self, event: PoseEvent):
        self._enqueue(event)

    def send_person_detected(self, event: PersonDetectedPayload):
        self._enqueue(event)

    def send_patient_pose(self, event: PatientPoseEvent):
        self._enqueue(event)

    def post_face_log(
        self,
        person_id:     str,
        name:          str,
        is_patient:    bool,
        confidence:    float,
        camera_id:     str  = "cam_0",
        recognized_at: float = 0.0,
    ) -> None:
        import time as _time
        self._enqueue(FaceLogPayload(
            person_id     = person_id,
            name          = name,
            is_patient    = is_patient,
            confidence    = max(0.0, min(1.0, confidence)),
            camera_id     = camera_id,
            recognized_at = recognized_at if recognized_at > 0 else _time.time(),
        ))

    def update_stats(self, fps: float, pose: PoseState):
        self.current_fps  = fps
        self.current_pose = pose

    # ── Blocking helpers ───────────────────────────────────────────────────────

    def fetch_config_sync(self) -> Optional[ThresholdConfig]:
        params = {"camera_id": self.camera_id}
        cfg = None
        try:
            r = httpx.get(f"{self.base_url}/config/thresholds", params=params, timeout=3.0)
            if r.status_code == 200:
                cfg = ThresholdConfig(**r.json())
        except Exception as e:
            self.last_error = str(e)
        try:
            r = httpx.get(f"{self.base_url}/config/features", params=params, timeout=3.0)
            if r.status_code == 200:
                self.current_features = FeatureConfig(**r.json())
        except Exception as e:
            self.last_error = str(e)
        return cfg

    def fetch_features_sync(self) -> Optional[FeatureConfig]:
        try:
            r = httpx.get(
                f"{self.base_url}/config/features",
                params={"camera_id": self.camera_id},
                timeout=3.0,
            )
            if r.status_code == 200:
                self.current_features = FeatureConfig(**r.json())
                return self.current_features
        except Exception as e:
            self.last_error = str(e)
        return None

    def fetch_family_members_sync(self) -> List[FamilyMember]:
        try:
            r = httpx.get(f"{self.base_url}/family-members", timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                rows = data if isinstance(data, list) else data.get("members", [])
                return [FamilyMember(**m) for m in rows]
        except Exception as e:
            self.last_error = str(e)
        return []

    def add_family_member_sync(self, payload: AddFamilyMemberPayload) -> bool:
        try:
            r = httpx.post(
                f"{self.base_url}/family-members/register",
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
                pass

    async def _check_health(self, client: httpx.AsyncClient):
        try:
            r = await client.get("/health")
            self.connected = r.status_code == 200
        except Exception as e:
            self.connected  = False
            self.last_error = str(e)

    async def _event_send_loop(self, client: httpx.AsyncClient):
        _ROUTES = {
            FallEvent:             "/events/fall",
            PoseEvent:             "/events/pose",
            PersonDetectedPayload: "/events/person-detected",
            PatientPoseEvent:      "/events/patient-pose",
            FaceLogPayload:        "/face-logs",
        }
        while self._running:
            try:
                event: _AnyEvent = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                if event is None:
                    break

                route = _ROUTES.get(type(event))
                if not route:
                    continue

                try:
                    body = _serialize(event)
                except Exception as e:
                    print(f"[BackendClient] serialize FAIL type={type(event).__name__} err={e!r}")
                    continue

                max_attempts = _MAX_ATTEMPTS.get(type(event), 1)
                timeout      = _TIMEOUT.get(type(event), 3.0)

                for attempt in range(max_attempts):
                    try:
                        r = await asyncio.wait_for(
                            client.post(route, content=body, headers=_JSON_HEADERS),
                            timeout=timeout,
                        )
                        self.connected = r.status_code < 500
                        if isinstance(event, PatientPoseEvent):
                            print(f"[BackendClient] patient-pose OK attempt={attempt} status={r.status_code}")
                        break
                    except Exception as e:
                        self.connected  = False
                        self.last_error = repr(e)
                        if isinstance(event, PatientPoseEvent):
                            print(f"[BackendClient] patient-pose FAIL attempt={attempt} type={type(e).__name__} err={e!r}")
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(1.0 * (attempt + 1))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.connected  = False
                self.last_error = repr(e)
                await asyncio.sleep(0.5)

    async def _config_poll_loop(self, client: httpx.AsyncClient):
        params = {"camera_id": self.camera_id}
        while self._running:
            await asyncio.sleep(self.config_interval)
            try:
                r = await client.get("/config/thresholds", params=params)
                if r.status_code == 200:
                    self.connected = True
                    if self.on_config_update:
                        self.on_config_update(ThresholdConfig(**r.json()))
            except Exception as e:
                self.connected  = False
                self.last_error = str(e)
            try:
                r = await client.get("/config/features", params=params)
                if r.status_code == 200:
                    feat = FeatureConfig(**r.json())
                    if feat != self.current_features:
                        self.current_features = feat
                        if self.on_feature_update:
                            self.on_feature_update(feat)
            except Exception as e:
                self.last_error = str(e)

    async def _heartbeat_loop(self, client: httpx.AsyncClient):
        while self._running:
            await asyncio.sleep(self.status_interval)
            payload = HeartbeatPayload(
                camera_id = self.camera_id,
                timestamp = time.time(),
                fps       = self.current_fps,
                state     = self.current_pose,
            )
            try:
                await client.post(
                    "/events/heartbeat",
                    content=payload.json().encode("utf-8"),
                    headers=_JSON_HEADERS,
                )
            except Exception:
                pass