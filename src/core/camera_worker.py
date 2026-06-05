"""
src/core/camera_worker.py — Camera capture thread
"""
from __future__ import annotations
import cv2
import os
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
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
from schemas import ThresholdConfig, FeatureConfig, PoseState, FallEvent, EventType

try:
    from .face_engine import FaceEngine, RecognizedPerson, FaceBox
    FACE_ENGINE_AVAILABLE = True
except (ImportError, Exception):
    FACE_ENGINE_AVAILABLE = False
    RecognizedPerson = object
    FaceBox          = object

try:
    from .appearance_tracker import TrackerManager
    TRACKER_AVAILABLE = True
except (ImportError, Exception):
    TRACKER_AVAILABLE = False

try:
    from .transformer_engine import TransformerEngine, TransformerResult
    TRANSFORMER_AVAILABLE = True
except (ImportError, Exception):
    TRANSFORMER_AVAILABLE = False
    TransformerResult = object

try:
    from .audio_engine import AudioEngine, AudioResult
    AUDIO_ENGINE_AVAILABLE = True
except (ImportError, Exception):
    AUDIO_ENGINE_AVAILABLE = False
    AudioResult = object



@dataclass
class PersonResult:
    person_id:    int
    bbox:         tuple
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
    all_person_results: List = field(default_factory=list)


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
        family_manager:  Optional[Any]             = None,
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
        self._family_manager = family_manager
        self._is_stream      = isinstance(camera_source, str) and "://" in camera_source
        self._running        = False
        self._paused         = False
        self._engine:         Optional[PoseEngine]        = None
        self._face_engine:    Optional[FaceEngine]        = None
        self._transformer:    Optional[TransformerEngine] = None
        self._audio_engine:   Optional[AudioEngine]       = None


        # [FIX] Mỗi person_id có pipeline riêng
        self._person_pipelines: dict = {}   # pid → DetectionPipeline
        self._person_last_seen: dict = {}   # pid → float timestamp
        self._PERSON_TIMEOUT          = 30.0

        self._fps_counter    = 0
        self._fps_timer      = time.time()
        self.fps             = 0.0
        self._last_frame:    Optional[np.ndarray] = None
        self._pending_audio: Optional[AudioResult] = None
        self._pending_audio_for_fall: Optional[object] = None

        self._tracker:             Optional[TrackerManager] = None
        self._frame_num:           int  = 0
        self._pending_recognition: dict = {}
        self._yolo_to_tracker:     dict = {}
        self._face_log_sent:       dict = {}

        self._persons_data_cache: list = []

        self._face_futures:       dict = {}
        self._face_executor:      Optional[Any] = None
        self._face_single_future: Optional[Any] = None
        self._face_single_cached: list = []

        self._frame_buffer: deque = deque(maxlen=300)

        self._post_remaining: int  = 0
        self._post_frames:    list = []
        self._pending_clip:   dict = {}

        self._clips_dir = Path(__file__).resolve().parent.parent.parent / "clips"
        self._clips_dir.mkdir(parents=True, exist_ok=True)
        self._clip_thread: Optional[threading.Thread] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def stop(self):   self._running = False
    def pause(self):  self._paused  = True
    def resume(self): self._paused  = False

    # ── Config / Feature update ────────────────────────────────────────────────

    def update_config(self, cfg: ThresholdConfig):
        self.config = cfg
        for pipeline in self._person_pipelines.values():
            pipeline.update_config(cfg)

    def update_features(self, feat: FeatureConfig):
        self.features = feat
        for pipeline in self._person_pipelines.values():
            pipeline.update_features(feat)

        # Bật/tắt audio engine theo feature
        if feat.enable_sound_detection:
            if self._audio_engine is None and AUDIO_ENGINE_AVAILABLE:
                try:
                    self._audio_engine = AudioEngine(
                        listen_seconds=feat.sound_listen_seconds,
                        on_result=self._on_audio_result,
                    )
                    print("[CameraWorker] AudioEngine khởi tạo từ feature update")
                except Exception as e:
                    print(f"[CameraWorker] AudioEngine init failed: {e}")
            elif self._audio_engine is not None:
                self._audio_engine.enabled = True
                self._audio_engine.listen_seconds = feat.sound_listen_seconds
        else:
            if self._audio_engine is not None:
                self._audio_engine.enabled = False

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset_falls(self):
        if self._transformer:
            self._transformer.reset()

        for pipeline in self._person_pipelines.values():
            pipeline.reset_falls()

    # ── Stats ──────────────────────────────────────────────────────────────────

    def fall_count(self) -> int:
        return sum(p.fall.count() for p in self._person_pipelines.values())

    def get_current_frame(self) -> Optional[np.ndarray]:
        return self._last_frame.copy() if self._last_frame is not None else None

    def audio_engine_status(self) -> str:
        if self._audio_engine is None:
            return "disabled"
        return self._audio_engine.status

    # ── Per-person pipeline ────────────────────────────────────────────────────

    def _get_or_create_pipeline(self, person_id: int) -> DetectionPipeline:
        """Mỗi person_id có DetectionPipeline riêng — state độc lập hoàn toàn."""
        if person_id not in self._person_pipelines:
            self._person_pipelines[person_id] = DetectionPipeline(
                config=self.config, features=self.features)
            print(f"[CameraWorker] New person tracked: ID={person_id}")
        self._person_last_seen[person_id] = time.time()
        return self._person_pipelines[person_id]

    def _cleanup_lost_persons(self):
        now  = time.time()
        lost = [pid for pid, t in self._person_last_seen.items()
                if now - t > self._PERSON_TIMEOUT]
        for pid in lost:
            self._person_pipelines.pop(pid, None)
            self._person_last_seen.pop(pid, None)
            t_id = self._yolo_to_tracker.pop(pid, None)
            if t_id is not None:
                self._pending_recognition.pop(t_id, None)
                self._face_log_sent.pop(t_id, None)
            print(f"[CameraWorker] Person lost: ID={pid}")

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _on_audio_result(self, result: "AudioResult"):
        self._pending_audio = result
        self._pending_audio_for_fall = result

    # ── Camera open ────────────────────────────────────────────────────────────

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            return None
        if self._is_stream:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif not isinstance(self.camera_source, str):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        self._engine = PoseEngine(
            use_yolo=self.use_yolo,
            model_complexity=self.config.model_complexity,
        )

        def _face_log(msg):
            print(f"[Face] {msg}")
            if self._family_manager and self._family_manager._on_log:
                self._family_manager._on_log(msg)

        _face_log(f"use_face={self.use_face} | face_ok={FACE_ENGINE_AVAILABLE} | feature_on={self.features.enable_face_recognition}")

        if self.use_face and self.features.enable_face_recognition and FACE_ENGINE_AVAILABLE:
            try:
                _face_log("Đang load FaceEngine (YOLOv8s + InsightFace)...")
                self._face_engine = FaceEngine()
                _face_log("✓ FaceEngine sẵn sàng")
                if self._family_manager is not None:
                    _face_log("Đang khởi động FamilyManager...")
                    self._family_manager.start(
                        shared_recognizer=self._face_engine._face_rec
                    )
                    self._face_engine.set_family_manager(self._family_manager)
                    _face_log("✓ FamilyManager đã khởi động")
                else:
                    _face_log("FamilyManager=None — dùng local face_db.pkl")
            except Exception as e:
                _face_log(f"✗ FaceEngine init lỗi: {e}")
        elif self.use_face and not FACE_ENGINE_AVAILABLE:
            _face_log("✗ ultralytics hoặc insightface chưa cài")

        if self._face_engine is not None and TRACKER_AVAILABLE:
            self._tracker = TrackerManager()
            _face_log("✓ AppearanceTracker initialized (face-once + appearance tracking)")
        elif self.use_face and not self.features.enable_face_recognition:
            _face_log("⚠ Feature enable_face_recognition=False từ backend — bị tắt")

        if self._face_engine is not None:
            self._face_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="FaceRecog")
            _face_log("✓ FaceRecognition background thread started")

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
                if isinstance(self.camera_source, str) and not self._is_stream:
                    print("[CameraWorker] Video file kết thúc")
                    # Chờ collect nốt post frames
                    while self._post_remaining > 0:
                        time.sleep(0.1)
                    # Chờ clip thread upload + gửi backend
                    if self._clip_thread is not None and self._clip_thread.is_alive():
                        print("[CameraWorker] Đang chờ gửi fall event...")
                        self._clip_thread.join(timeout=30)
                    break
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

            if isinstance(self.camera_source, str) and not self._is_stream:
                fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
                time.sleep(1.0 / fps_src)

            if self.config.flip_horizontal:
                frame = cv2.flip(frame, 1)
            self._last_frame = frame

            self._frame_num += 1
            face_active = (self._face_engine is not None and
                           self.features.enable_face_recognition)

            # ── Phase 0: Thu kết quả nhận diện async ──────────────────────
            if face_active and self._tracker is not None and self._face_executor is not None:
                done_tids = [tid for tid, fut in list(self._face_futures.items())
                             if fut.done()]
                for tid in done_tids:
                    fut = self._face_futures.pop(tid)
                    try:
                        p_info = fut.result()
                    except Exception:
                        p_info = None
                    self._pending_recognition.pop(tid, None)
                    if p_info is not None and p_info.is_known:
                        self._tracker.update_track_identity(
                            tid, p_info.person_id, p_info.name,
                            p_info.is_patient, p_info.notify_on_fall)
                        print(f"[Tracker] Identified (async): {p_info.name}")
                        if self._face_log_sent.get(tid) != p_info.person_id:
                            self._face_log_sent[tid] = p_info.person_id
                            if self._backend_client is not None:
                                self._backend_client.post_face_log(
                                    person_id  = p_info.person_id,
                                    name       = p_info.name,
                                    is_patient = p_info.is_patient,
                                    confidence = p_info.confidence,
                                    camera_id  = self.camera_id,
                                )

            self._cleanup_lost_persons()

            # Process mỗi 2 frame để giảm tải
            # if self._frame_num % 2 == 0:
            #     persons_data = self._engine.process_multi(frame)
            #     self._persons_data_cache = persons_data
            # else:
            #     persons_data = getattr(self, "_persons_data_cache", [])
            if self._frame_num % 2 == 0:
                persons_data = self._engine.process_multi(frame)
                self._persons_data_cache = persons_data
                self._should_process = True
            else:
                persons_data = getattr(self, "_persons_data_cache", [])
                self._should_process = False

            all_person_results: List[PersonResult] = []
            result       = DetectionResult(state=PoseState.UNKNOWN)
            trans_result = None

            if not persons_data:
                # ── [FIX] Không fallback MediaPipe nữa
                # YOLO không detect được ai → bỏ qua frame, không trigger fall
                result = DetectionResult(state=PoseState.UNKNOWN)

            else:
                # ── Multi-person: mỗi người dùng pipeline riêng ─────────────
                sit_prob = 0.0
                persons_data = sorted(persons_data, key=lambda pd: pd.metrics.confidence, reverse=True)[:1]

                for pd in persons_data:
                    # [FIX] Pipeline riêng per-person
                    person_pipeline = self._get_or_create_pipeline(pd.person_id)
                    # person_result   = person_pipeline.process(
                    #     pd.metrics, sitting_prob=sit_prob
                    # )
                    if getattr(self, '_should_process', True):
                        person_result = person_pipeline.process(
                            pd.metrics, sitting_prob=sit_prob
                        )
                        person_pipeline.last_result = person_result
                    else:
                        person_result = getattr(person_pipeline, 'last_result',
                                                person_pipeline.process(pd.metrics, sitting_prob=sit_prob))

                    person_trans = None
                    if self._transformer is not None:
                        self._transformer.push(pd.raw_kp)
                        person_trans = self._transformer.get()

                    all_person_results.append(PersonResult(
                        person_id    = pd.person_id,
                        bbox         = pd.bbox,
                        result       = person_result,
                        trans_result = person_trans,
                    ))

                    # Appearance tracking + face recognition
                    if face_active and self._tracker is not None:
                        box      = pd.bbox
                        track_id = self._tracker.match_box(box, self._frame_num)

                        if track_id is None:
                            if self._tracker.has_track(pd.person_id):
                                track_id = pd.person_id
                                self._tracker.update_last_box(track_id, box, self._frame_num)
                            else:
                                track_id = pd.person_id
                                self._tracker.add_track(
                                    track_id, None, "", False, None,
                                    box, self._frame_num,
                                )
                                print(f"[Tracker] New person detected: track_id={track_id}")
                        self._yolo_to_tracker[pd.person_id] = track_id

                        if not self._tracker.is_identified(track_id):
                            if (self._face_executor is not None
                                    and track_id not in self._face_futures):
                                last_tried = self._pending_recognition.get(track_id, -100)
                                if self._frame_num - last_tried >= 5:
                                    self._pending_recognition[track_id] = self._frame_num
                                    self._face_futures[track_id] = (
                                        self._face_executor.submit(
                                            self._face_engine.run_face_recognition,
                                            frame.copy(), box,
                                        )
                                    )
                        else:
                            if self._frame_num % 10 == 0:
                                try:
                                    snap = self._face_engine.extract_appearance(frame, box)
                                except Exception:
                                    snap = None
                                if snap is not None:
                                    reason = self._tracker._reidentify_reason(track_id, snap)
                                    if (reason and self._face_executor is not None
                                            and track_id not in self._face_futures):
                                        t_info = self._tracker.get_track_info(track_id)
                                        t_name = t_info.get("name", "?") if t_info else "?"
                                        print(f"[Tracker] Re-ID triggered ({reason}): {t_name}")
                                        self._face_futures[track_id] = (
                                            self._face_executor.submit(
                                                self._face_engine.run_face_recognition,
                                                frame.copy(), box,
                                            )
                                        )
                                    self._tracker.update_snapshot(track_id, snap, self._frame_num)

                if self._tracker is not None:
                    self._tracker.remove_stale(self._frame_num, max_gap=60)

                # Chọn result đại diện
                fall_persons = [r for r in all_person_results if r.result.is_falling]
                if fall_persons:
                    worst        = max(fall_persons, key=lambda r: r.result.fall_max_velocity)
                    result       = worst.result
                    trans_result = worst.trans_result
                elif all_person_results:
                    result       = all_person_results[0].result
                    trans_result = all_person_results[0].trans_result

                any_fall_triggered = any(
                    r.result.fall_just_triggered for r in all_person_results
                )
                if any_fall_triggered and self._audio_engine is not None:
                    self._audio_engine.trigger()

            # ── Face recognition ───────────────────────────────────────────
            persons = []
            if face_active:
                if self._tracker is not None and all_person_results:
                    for pr in all_person_results:
                        t_key  = self._yolo_to_tracker.get(pr.person_id, pr.person_id)
                        t_info = self._tracker.get_track_info(t_key)
                        if not (t_info and t_info["person_id"]):
                            continue
                        pid    = t_info["person_id"]
                        fm_info = None
                        if self._family_manager is not None:
                            fm_info = self._family_manager.get_member_info(pid)
                        name           = fm_info["name"]           if fm_info else t_info["name"]
                        is_patient     = fm_info["is_patient"]     if fm_info else t_info["is_patient"]
                        notify_on_fall = fm_info["notify_on_fall"] if fm_info else t_info.get("notify_on_fall", True)
                        x1, y1, x2, y2 = pr.bbox
                        persons.append(RecognizedPerson(
                            person_id      = pid,
                            name           = name,
                            confidence     = 1.0,
                            box            = FaceBox(x1, y1, x2, y2),
                            is_known       = True,
                            is_patient     = is_patient,
                            notify_on_fall = notify_on_fall,
                        ))
                elif self._face_engine is not None:
                    if (self._face_single_future is not None
                            and self._face_single_future.done()):
                        try:
                            _res = self._face_single_future.result()
                            if isinstance(_res, list):
                                self._face_single_cached = _res
                        except Exception:
                            pass
                        self._face_single_future = None
                    persons = list(self._face_single_cached)
                    if (self._face_executor is not None
                            and self._face_single_future is None
                            and self._frame_num % 30 == 0):
                        self._face_single_future = self._face_executor.submit(
                            self._face_engine.process_frame, frame.copy()
                        )

            # ── FPS ───────────────────────────────────────────────────────
            self._fps_counter += 1
            now = time.time()
            if now - self._fps_timer >= 1.0:
                self.fps          = self._fps_counter / (now - self._fps_timer)
                self._fps_counter = 0
                self._fps_timer   = now

            # ── Audio result ──────────────────────────────────────────────
            audio_res = self._pending_audio
            if audio_res is not None:
                self._pending_audio = None

            # ── Rolling buffer ─────────────────────────────────────────────
            f = frame.copy()
            self._frame_buffer.append(f)

            # ── Post-fall clip collection ──────────────────────────────────
            if self._post_remaining > 0:
                self._post_frames.append(f)
                self._post_remaining -= 1
                if self._post_remaining == 0:
                    self._clip_thread = threading.Thread(
                        target=self._process_fall_clip,
                        args=(
                            self._pending_clip.get("pre_frames", []),
                            list(self._post_frames),
                            self._pending_clip.get("fall_data", {}),
                        ),
                        daemon=False,
                    )
                    self._clip_thread.start()
                    self._post_frames = []
                    self._pending_clip = {}

            # ── Fall trigger → clip capture ────────────────────────────────
            _fall_triggered = result.fall_just_triggered or (
                bool(all_person_results) and
                any(r.result.fall_just_triggered for r in all_person_results)
            )
            if _fall_triggered and self._post_remaining == 0:
                self._pending_audio_for_fall = None
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

            # ── Đưa vào queue ─────────────────────────────────────────────
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
        if self._face_executor is not None:
            self._face_executor.shutdown(wait=False)

        # Chờ clip thread gửi xong trước khi đóng
        if self._clip_thread is not None and self._clip_thread.is_alive():
            print("[CameraWorker] Chờ clip thread hoàn tất...")
            self._clip_thread.join(timeout=60)

    # ── Clip save / upload / send ──────────────────────────────────────────────
    def _process_fall_clip(self, pre_frames: list, post_frames: list, fall_data: dict):
        all_frames = pre_frames + post_frames
        if not all_frames:
            return

        fps = 30
        take = 3 * fps
        frames = all_frames[:take] + all_frames[-take:]

        audio_res = None

        # ── Gửi fall event NGAY (không có clip) ──────────────────────────────
        event_id = None
        if self._backend_client is not None:
            try:
                import httpx as _httpx
                event = FallEvent(
                    event_type=EventType.FALL,
                    camera_id=self.camera_id,
                    timestamp=fall_data.get("timestamp", time.time()),
                    state=PoseState.FALLING,
                    state_before=fall_data.get("state_before", PoseState.UNKNOWN),
                    velocity_px_per_s=fall_data.get("velocity_px_per_s", 0.0),
                    max_velocity=fall_data.get("max_velocity", 0.0),
                    body_angle=fall_data.get("body_angle", 0.0),
                    confidence=fall_data.get("confidence", 0.0),
                    clip_url=None,  # chưa có clip
                    sound_detected=False,
                    sound_class="",
                    sound_confidence=0.0,
                )
                r = _httpx.post(
                    f"{self._backend_client.base_url}/events/fall",
                    content=event.json().encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    timeout=10.0,
                )
                if r.status_code in (200, 201):
                    event_id = r.json().get("id")
                    print(f"[CameraWorker] FallEvent sent immediately — id={event_id}")
                else:
                    print(f"[CameraWorker] FallEvent send failed status={r.status_code}")
            except Exception as exc:
                print(f"[CameraWorker] FallEvent immediate send failed: {exc}")

        # ── Ghi clip ──────────────────────────────────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._clips_dir / f"fall_{ts}.mp4"
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 30, (w, h))
        for frm in frames:
            writer.write(frm)
        writer.release()
        print(f"[CameraWorker] Clip saved: {path}  ({len(frames)} frames)")

        # ── Upload Cloudinary ─────────────────────────────────────────────────
        clip_url = None
        if _CLOUDINARY_OK:
            try:
                res = cloudinary.uploader.upload(
                    str(path),
                    resource_type="video",
                    folder="fall_detection",
                    public_id=f"fall_{ts}",
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

        # ── Cập nhật clip_url vào event đã gửi ───────────────────────────────
        if clip_url and event_id and self._backend_client is not None:
            try:
                import httpx as _httpx
                r = _httpx.patch(
                    f"{self._backend_client.base_url}/events/fall/{event_id}",
                    json={"clip_url": clip_url},
                    timeout=10.0,
                )
                print(f"[CameraWorker] Clip URL updated — event_id={event_id} status={r.status_code}")
            except Exception as exc:
                print(f"[CameraWorker] Clip URL update failed: {exc}")