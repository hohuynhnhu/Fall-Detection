"""
src/core/camera_worker.py — Camera capture thread
"""
from __future__ import annotations
import cv2
import os
import threading
import queue
import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List
from collections import defaultdict, deque

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

try:
    import cloudinary
    import cloudinary.uploader
    cloudinary.config(
        cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME", ""),
        api_key    = os.environ.get("CLOUDINARY_API_KEY", ""),
        api_secret = os.environ.get("CLOUDINARY_API_SECRET", ""),
    )
    _CLOUDINARY_OK = bool(os.environ.get("CLOUDINARY_CLOUD_NAME", ""))
except ImportError:
    _CLOUDINARY_OK = False

from .pose_engine import PoseEngine
from .fall_detector import DetectionPipeline, DetectionResult
from .overlay import STATE_SKELETON_COLORS
from    schemas import ThresholdConfig, FeatureConfig, PoseState, FallEvent, EventType

try:
    from .face_engine import FaceEngine, RecognizedPerson
    FACE_ENGINE_AVAILABLE = True
except (ImportError, Exception):
    FACE_ENGINE_AVAILABLE = False
    RecognizedPerson = object  # type: ignore

try:
    from .transformer_engine import TransformerEngine, TransformerResult
    TRANSFORMER_AVAILABLE = True
except (ImportError, Exception):
    TRANSFORMER_AVAILABLE = False
    TransformerResult = object  # type: ignore

try:
    from .audio_engine import AudioEngine, AudioResult
    AUDIO_ENGINE_AVAILABLE = True
except (ImportError, Exception):
    AUDIO_ENGINE_AVAILABLE = False
    AudioResult = object  # type: ignore


@dataclass
class PersonResult:
    person_id:    int
    bbox:         tuple           # (x1, y1, x2, y2)
    result:       DetectionResult
    trans_result: Optional[object] = None


@dataclass
class WorkerFrame:
    frame_bgr:          np.ndarray
    result:             DetectionResult
    fps:                float
    timestamp:          float
    recognized_persons: List = field(default_factory=list)
    transformer_result: Optional[object] = None
    audio_result:       Optional[object] = None
    all_person_results: List = field(default_factory=list)  # multi-person


class CameraWorker(threading.Thread):
    def __init__(
        self,
        camera_source:   "int | str" = 0,
        result_queue:    Optional[queue.Queue] = None,
        use_yolo:        bool = False,
        use_face:        bool = False,
        use_transformer: bool = False,
        config:          Optional[ThresholdConfig] = None,
        features:        Optional[FeatureConfig]   = None,
        camera_id:       str                       = "cam_0",
        backend_client:  Optional[Any]             = None,
    ):
        super().__init__(daemon=True)
        self.camera_source   = camera_source
        self.result_queue    = result_queue or queue.Queue(maxsize=3)
        self.use_yolo        = use_yolo
        self.use_face        = use_face
        self.use_transformer = use_transformer
        self.config          = config   or ThresholdConfig()
        self.features        = features or FeatureConfig()
        self.camera_id       = camera_id
        self._backend_client = backend_client
        self._is_stream      = isinstance(camera_source, str) and "://" in camera_source
        self._running        = False
        self._paused         = False
        self._engine:        Optional[PoseEngine]        = None
        self._pipeline:      Optional[DetectionPipeline] = None
        self._face_engine:   Optional[FaceEngine]        = None
        self._transformer:   Optional[TransformerEngine] = None
        self._audio_engine:  Optional[AudioEngine]       = None

        # Multi-person tracking
        self._person_pipelines:    dict = {}   # {person_id: DetectionPipeline}
        self._person_transformers: dict = {}   # {person_id: TransformerEngine}
        self._person_last_seen:    dict = {}   # {person_id: timestamp}
        self._PERSON_TIMEOUT             = 5.0

        self._fps_counter    = 0
        self._fps_timer      = time.time()
        self.fps             = 0.0
        self._last_frame:    Optional[np.ndarray] = None
        self._pending_audio: Optional[AudioResult] = None

        # Rolling buffer: 300 frames = 10 s @ 30 fps
        self._frame_buffer: deque = deque(maxlen=300)

        # Post-fall clip state
        self._post_remaining: int  = 0   # frames left to collect after trigger
        self._post_frames:    list = []
        self._pending_clip:   dict = {}  # pre_frames + fall_data

        # Clips output directory (project_root/clips/)
        self._clips_dir = Path(__file__).resolve().parent.parent.parent / "clips"
        self._clips_dir.mkdir(parents=True, exist_ok=True)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def stop(self):   self._running = False
    def pause(self):  self._paused  = True
    def resume(self): self._paused  = False

    # ── Config / Feature update ────────────────────────────────────────────────

    def update_config(self, cfg: ThresholdConfig):
        self.config = cfg
        if self._pipeline:
            self._pipeline.update_config(cfg)
        # Cập nhật config cho tất cả pipeline của từng người
        for pipeline in self._person_pipelines.values():
            pipeline.update_config(cfg)

    def update_features(self, feat: FeatureConfig):
        self.features = feat
        if self._pipeline:
            self._pipeline.update_features(feat)
        if self._audio_engine is not None:
            self._audio_engine.enabled        = feat.enable_sound_detection
            self._audio_engine.listen_seconds = feat.sound_listen_seconds

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset_falls(self):
        """Reset fall history cho tất cả người + fallback pipeline."""
        if self._pipeline:
            self._pipeline.reset_falls()
        if self._transformer:
            self._transformer.reset()
        for pipeline in self._person_pipelines.values():
            pipeline.reset_falls()
        for te in self._person_transformers.values():
            te.reset()

    # ── Stats ──────────────────────────────────────────────────────────────────

    def fall_count(self) -> int:
        """Tổng số lần té ngã của tất cả người được track."""
        total = sum(p.fall.count() for p in self._person_pipelines.values())
        # Fallback về pipeline đơn nếu không có multi-person
        return total if total > 0 else (
            self._pipeline.fall.count() if self._pipeline else 0)

    def get_current_frame(self) -> Optional[np.ndarray]:
        return self._last_frame.copy() if self._last_frame is not None else None

    def audio_engine_status(self) -> str:
        if self._audio_engine is None:
            return "disabled"
        return self._audio_engine.status

    # ── Multi-person helpers ───────────────────────────────────────────────────

    def _get_or_create_person(self, person_id: int):
        """Tạo pipeline + transformer riêng cho mỗi person_id nếu chưa có."""
        if person_id not in self._person_pipelines:
            self._person_pipelines[person_id] = DetectionPipeline(
                config=self.config, features=self.features)
            print(f"[CameraWorker] New person tracked: ID={person_id}")

            if self.use_transformer and TRANSFORMER_AVAILABLE:
                te = TransformerEngine()
                if te.loaded:
                    self._person_transformers[person_id] = te

        self._person_last_seen[person_id] = time.time()

    def _cleanup_lost_persons(self):
        """Xóa người không còn trong frame sau PERSON_TIMEOUT giây."""
        now  = time.time()
        lost = [pid for pid, t in self._person_last_seen.items()
                if now - t > self._PERSON_TIMEOUT]
        for pid in lost:
            self._person_pipelines.pop(pid, None)
            self._person_transformers.pop(pid, None)
            self._person_last_seen.pop(pid, None)
            print(f"[CameraWorker] Person lost: ID={pid}")

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _on_audio_result(self, result: "AudioResult"):
        self._pending_audio = result

    # ── Camera open ────────────────────────────────────────────────────────────

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            return None
        if self._is_stream:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        # Khởi tạo engines
        self._engine   = PoseEngine(
            use_yolo=self.use_yolo,
            model_complexity=self.config.model_complexity,
        )
        self._pipeline = DetectionPipeline(config=self.config, features=self.features)

        if self.use_face and self.features.enable_face_recognition and FACE_ENGINE_AVAILABLE:
            try:
                self._face_engine = FaceEngine()
            except Exception as e:
                print(f"[CameraWorker] FaceEngine init failed: {e}")

        if self.use_transformer and TRANSFORMER_AVAILABLE:
            try:
                self._transformer = TransformerEngine()
                if not self._transformer.loaded:
                    self._transformer = None
            except Exception as e:
                print(f"[CameraWorker] TransformerEngine init failed: {e}")

        if self.features.enable_sound_detection and AUDIO_ENGINE_AVAILABLE:
            try:
                self._audio_engine = AudioEngine(
                    listen_seconds=self.features.sound_listen_seconds,
                    on_result=self._on_audio_result,
                )
            except Exception as e:
                print(f"[CameraWorker] AudioEngine init failed: {e}")

        cap = self._open_capture()
        if cap is None:
            self.result_queue.put({"error": f"Không mở được nguồn: {self.camera_source}"})
            return

        self._running    = True
        consecutive_fail = 0

        while self._running:
            if self._paused:
                time.sleep(0.03)
                continue

            ret, frame = cap.read()
            if not ret:
                consecutive_fail += 1
                if self._is_stream and consecutive_fail <= 10:
                    print(f"[CameraWorker] Stream mất kết nối, thử lại ({consecutive_fail}/10)…")
                    cap.release()
                    time.sleep(1.0)
                    cap = self._open_capture() or cap
                    continue
                time.sleep(0.01)
                continue

            consecutive_fail = 0
            if self.config.flip_horizontal:
                frame = cv2.flip(frame, 1)
            self._last_frame = frame

            # ── Xử lý nhiều người ─────────────────────────────────────────────
            self._cleanup_lost_persons()
            persons_data     = self._engine.process_multi(frame)
            all_person_results: List[PersonResult] = []
            result       = DetectionResult(state=PoseState.UNKNOWN)
            trans_result = None

            if not persons_data:
                # Fallback: không có YOLO tracking → xử lý 1 người
                metrics, landmarks, raw_kp = self._engine.process(frame)
                if metrics is not None:
                    result = self._pipeline.process(metrics)
                else:
                    result = DetectionResult(state=PoseState.UNKNOWN)
                if self._transformer is not None:
                    self._transformer.push(raw_kp)
                    trans_result = self._transformer.get()
                if landmarks:
                    color = STATE_SKELETON_COLORS.get(str(result.state), (80, 220, 80))
                    self._engine.draw_skeleton(frame, landmarks, color)

            else:
                # Multi-person: xử lý từng người riêng
                for pd in persons_data:
                    self._get_or_create_person(pd.person_id)

                    # Rule-based riêng từng người
                    pipeline      = self._person_pipelines[pd.person_id]
                    person_result = pipeline.process(pd.metrics)

                    # Transformer riêng từng người
                    person_trans = None
                    if pd.person_id in self._person_transformers:
                        te = self._person_transformers[pd.person_id]
                        te.push(pd.raw_kp)
                        person_trans = te.get()

                    # Vẽ bbox + ID + state lên frame
                    color = STATE_SKELETON_COLORS.get(
                        str(person_result.state), (80, 220, 80))
                    x1, y1, x2, y2 = pd.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID:{pd.person_id} {person_result.state}",
                        (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                    )

                    all_person_results.append(PersonResult(
                        person_id    = pd.person_id,
                        bbox         = pd.bbox,
                        result       = person_result,
                        trans_result = person_trans,
                    ))

                # Ưu tiên người đang té để hiển thị lên UI
                fall_persons = [r for r in all_person_results
                                if r.result.is_falling]
                if fall_persons:
                    # Lấy người có velocity cao nhất
                    worst        = max(fall_persons,
                                       key=lambda r: r.result.fall_max_velocity)
                    result       = worst.result
                    trans_result = worst.trans_result
                elif all_person_results:
                    result       = all_person_results[0].result
                    trans_result = all_person_results[0].trans_result

                # Trigger audio 1 lần nếu có bất kỳ người nào té
                any_fall_triggered = any(
                    r.result.fall_just_triggered for r in all_person_results)
                if any_fall_triggered and self._audio_engine is not None:
                    self._audio_engine.trigger()

            # ── Face recognition ───────────────────────────────────────────────
            persons    = []
            face_active = (self._face_engine is not None and
                           self.features.enable_face_recognition)
            if face_active:
                try:
                    persons = self._face_engine.process(frame)
                except Exception:
                    pass

            # ── FPS ───────────────────────────────────────────────────────────
            self._fps_counter += 1
            now = time.time()
            if now - self._fps_timer >= 1.0:
                self.fps          = self._fps_counter / (now - self._fps_timer)
                self._fps_counter = 0
                self._fps_timer   = now

            # ── Audio result ───────────────────────────────────────────────────
            audio_res = self._pending_audio
            if audio_res is not None:
                self._pending_audio = None

            # ── Rolling buffer (push rendered frame) ───────────────────────
            f = frame.copy()
            self._frame_buffer.append(f)

            # ── Post-fall frame collection ─────────────────────────────────
            if self._post_remaining > 0:
                self._post_frames.append(f)
                self._post_remaining -= 1
                if self._post_remaining == 0:
                    threading.Thread(
                        target=self._process_fall_clip,
                        args=(
                            self._pending_clip.get("pre_frames", []),
                            list(self._post_frames),
                            self._pending_clip.get("fall_data", {}),
                        ),
                        daemon=True,
                    ).start()
                    self._post_frames = []
                    self._pending_clip = {}

            # ── Fall trigger → start clip capture ─────────────────────────
            _fall_triggered = result.fall_just_triggered or (
                bool(all_person_results) and
                any(r.result.fall_just_triggered for r in all_person_results)
            )
            if _fall_triggered and self._post_remaining == 0:
                pre = list(self._frame_buffer)
                m   = result.metrics
                self._pending_clip = {
                    "pre_frames": pre[-150:] if len(pre) > 150 else pre,
                    "fall_data": {
                        "timestamp":         now,
                        "state_before":      result.prev_state,
                        "velocity_px_per_s": result.velocity_y,
                        "max_velocity":      result.fall_max_velocity,
                        "body_angle":        m.body_angle  if m else 0.0,
                        "confidence":        m.confidence  if m else 0.0,
                    },
                }
                self._post_remaining = 90
                self._post_frames    = []
                print(
                    f"[CameraWorker] Fall clip: {len(self._pending_clip['pre_frames'])} "
                    "pre + 90 post frames queued"
                )

            # ── Đưa vào queue ─────────────────────────────────────────────────
            payload = WorkerFrame(
                frame_bgr          = frame,
                result             = result,
                fps                = self.fps,
                timestamp          = now,
                recognized_persons = persons,
                transformer_result = trans_result,
                audio_result       = audio_res,
                all_person_results = all_person_results,
            )
            try:
                self.result_queue.put_nowait(payload)
            except queue.Full:
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.put_nowait(payload)
                except Exception:
                    pass

        cap.release()
        if self._engine:
            self._engine.close()

    # ── Clip save / upload / send ──────────────────────────────────────────────

    def _process_fall_clip(
        self,
        pre_frames: list,
        post_frames: list,
        fall_data: dict,
    ):
        """Chạy trong thread riêng: lưu MP4 → upload Cloudinary → POST /events/fall."""
        frames = pre_frames + post_frames
        if not frames:
            return

        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        path   = self._clips_dir / f"fall_{ts}.mp4"
        h, w   = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 30, (w, h))
        for frm in frames:
            writer.write(frm)
        writer.release()
        print(f"[CameraWorker] Clip saved: {path}  ({len(frames)} frames)")

        # Upload Cloudinary
        clip_url = ""
        if _CLOUDINARY_OK:
            try:
                res = cloudinary.uploader.upload(
                    str(path),
                    resource_type = "video",
                    folder        = "fall_detection",
                    public_id     = f"fall_{ts}",
                )
                clip_url = res.get("secure_url", "")
                print(f"[CameraWorker] Clip uploaded: {clip_url}")
            except Exception as exc:
                print(f"[CameraWorker] Cloudinary upload failed: {exc}")
            finally:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            print("[CameraWorker] Cloudinary not configured — clip kept local")

        # POST /events/fall
        if self._backend_client is not None:
            try:
                event = FallEvent(
                    event_type        = EventType.FALL,
                    camera_id         = self.camera_id,
                    timestamp         = fall_data.get("timestamp", time.time()),
                    state             = PoseState.FALLING,
                    state_before      = fall_data.get("state_before", PoseState.UNKNOWN),
                    velocity_px_per_s = fall_data.get("velocity_px_per_s", 0.0),
                    max_velocity      = fall_data.get("max_velocity", 0.0),
                    body_angle        = fall_data.get("body_angle", 0.0),
                    confidence        = fall_data.get("confidence", 0.0),
                    clip_url          = clip_url or None,
                )
                self._backend_client.send_fall(event)
                print(f"[CameraWorker] FallEvent sent (clip_url={'set' if clip_url else 'none'})")
            except Exception as exc:
                print(f"[CameraWorker] Backend send_fall failed: {exc}")