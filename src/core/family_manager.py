"""
src/core/family_manager.py — Quản lý danh sách thành viên gia đình từ backend.

Flow:
  1. start() → khởi tạo InsightFace → GET /family-members/all → download ảnh → extract embedding
  2. WebSocket /ws/desktop → lắng nghe new_member / remove_member real-time
  3. FaceEngine gọi match_person(embedding_512d) để nhận diện (cosine similarity)
"""
from __future__ import annotations
import asyncio
import json
import logging
import threading
import cv2
import numpy as np
import requests
from typing import Dict, List, Optional

try:
    import websockets
    _WS_OK = True
except ImportError:
    _WS_OK = False

try:
    from .face_recognizer_if import FaceRecognizer, cosine_similarity
    _FACE_OK = True
except ImportError:
    _FACE_OK = False

logger = logging.getLogger(__name__)

RECOGNITION_THRESHOLD = 0.35   # cosine similarity


class FamilyManager:
    """
    Manages 512-d ArcFace embeddings of family members loaded from backend.
    Thread-safe: dùng threading.Lock khi đọc/ghi dict _members.
    InsightFace buffalo_l được khởi tạo nội bộ trong start().
    """

    def __init__(
        self,
        base_url:  str,
        threshold: float = RECOGNITION_THRESHOLD,
        on_log:    Optional[callable] = None,
    ):
        self._base_url  = base_url.rstrip("/")
        self._threshold = threshold
        self._on_log    = on_log   # callback(msg: str) hiện log lên UI
        # {person_id: {name, is_patient, embeddings: [ndarray(512)]}}
        self._members: Dict[str, dict] = {}
        self._lock     = threading.Lock()
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._face:    Optional[FaceRecognizer]   = None
        self._match_log_cnt = 0

    # ── Public ──────────────────────────────────────────────────────────────────

    def start(self):
        """
        Khởi tạo InsightFace, load gallery từ API, bắt đầu WS listener thread.
        Gọi từ CameraWorker.run() sau khi FaceEngine đã sẵn sàng.
        """
        if not _FACE_OK:
            self._log("⚠ FamilyManager: insightface chưa cài — tắt nhận diện")
            return
        try:
            self._face = FaceRecognizer()
        except Exception as e:
            self._log(f"✗ FamilyManager: FaceRecognizer init lỗi: {e}")
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="FamilyManager"
        )
        self._thread.start()

    def stop(self):
        self._running = False

    def match_person(self, embedding: np.ndarray) -> Optional[dict]:
        """
        Cosine similarity match với toàn bộ thành viên đã load.
        Trả về {person_id, name, is_patient, confidence} hoặc None nếu không khớp.
        """
        best_id    = None
        best_score = -1.0
        with self._lock:
            n_members = len(self._members)
            for pid, data in self._members.items():
                for emb in data["embeddings"]:
                    score = cosine_similarity(embedding, emb)
                    if score > best_score:
                        best_score = score
                        best_id    = pid

        self._match_log_cnt += 1
        if self._match_log_cnt <= 3 or self._match_log_cnt % 100 == 0:
            logger.debug(
                "[FamilyManager] match #%d: members=%d best_score=%.3f threshold=%.3f",
                self._match_log_cnt, n_members, best_score, self._threshold,
            )

        if best_id is None or best_score < self._threshold:
            return None
        with self._lock:
            d = self._members[best_id]
            return {
                "person_id":  best_id,
                "name":       d["name"],
                "is_patient": d["is_patient"],
                "confidence": float(best_score),
            }

    def add_encoding(
        self,
        person_id:  str,
        name:       str,
        is_patient: bool,
        embedding:  np.ndarray,
    ):
        """Thêm embedding 512-d trực tiếp — dùng khi đăng ký khuôn mặt qua UI desktop."""
        with self._lock:
            if person_id not in self._members:
                self._members[person_id] = {
                    "name":       name,
                    "is_patient": is_patient,
                    "embeddings": [],
                }
            self._members[person_id]["embeddings"].append(embedding)

    def remove(self, person_id: str):
        with self._lock:
            self._members.pop(person_id, None)

    def member_count(self) -> int:
        with self._lock:
            return len(self._members)

    # ── Internal ────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            self._on_log(msg)

    def _run(self):
        """Entry point của daemon thread: load API rồi chạy WS loop."""
        self._load_from_api()
        if not _WS_OK:
            self._log("⚠ Face: websockets chưa cài — WS listener bị tắt")
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_loop())
        finally:
            loop.close()

    def _load_from_api(self):
        """GET /family-members/all → download ảnh từ URL → InsightFace extract embedding."""
        self._log(f"Face: GET {self._base_url}/family-members/all ...")
        try:
            r = requests.get(f"{self._base_url}/family-members/all", timeout=5)
            if r.status_code != 200:
                self._log(f"⚠ Face: API trả HTTP {r.status_code} — bỏ qua")
                return
            data = r.json()
            members: List[dict] = data if isinstance(data, list) else data.get("members", [])
            # Bỏ qua thành viên chưa có ảnh (chưa đăng ký từ mobile)
            members = [m for m in members if m.get("face_image_url")]
            if not members:
                self._log("Face: không có thành viên nào có ảnh trong DB")
                return
            self._log(f"Face: tải {len(members)} thành viên, đang trích xuất đặc trưng...")
            for m in members:
                self._download_and_encode(
                    url        = m["face_image_url"],
                    person_id  = m["person_id"],
                    name       = m["name"],
                    is_patient = m.get("is_patient", False),
                )
            self._log(f"✓ Face: sẵn sàng nhận diện {self.member_count()} thành viên")
        except Exception as e:
            self._log(f"✗ Face: load API thất bại: {e}")

    def _download_and_encode(
        self, url: str, person_id: str, name: str, is_patient: bool
    ):
        """Download ảnh từ URL, InsightFace extract 512-d embedding, lưu vào gallery."""
        try:
            if self._face is None:
                raise RuntimeError("FaceRecognizer chưa khởi tạo")
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            arr   = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("imdecode thất bại — ảnh không hợp lệ")
            emb = self._face.get_embedding_from_image(frame)
            if emb is None:
                raise ValueError("không phát hiện khuôn mặt trong ảnh")
            with self._lock:
                if person_id not in self._members:
                    self._members[person_id] = {
                        "name":       name,
                        "is_patient": is_patient,
                        "embeddings": [],
                    }
                self._members[person_id]["embeddings"].append(emb)
            tag = " [BN]" if is_patient else ""
            self._log(f"  ✓ {name}{tag} (id={person_id})")
        except Exception as e:
            self._log(f"  ✗ {name}: {e}")

    async def _ws_loop(self):
        """Kết nối WebSocket /ws/desktop, tự reconnect khi mất kết nối."""
        ws_url = (
            self._base_url
            .replace("http://", "ws://")
            .replace("https://", "wss://")
        ) + "/ws/desktop"
        while self._running:
            try:
                async with websockets.connect(
                    ws_url, ping_interval=20, open_timeout=5
                ) as ws:
                    self._log(f"Face WS: kết nối {ws_url}")
                    while self._running:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            self._handle_ws_message(json.loads(raw))
                        except asyncio.TimeoutError:
                            continue
            except Exception as e:
                self._log(f"Face WS: mất kết nối ({e}) — thử lại 5s")
                await asyncio.sleep(5.0)

    def _handle_ws_message(self, msg: dict):
        msg_type = msg.get("type")
        if msg_type == "new_member":
            self._log(f"Face WS: thành viên mới '{msg.get('name', '')}'")
            self._download_and_encode(
                url        = msg["face_image_url"],
                person_id  = msg["person_id"],
                name       = msg.get("name", ""),
                is_patient = msg.get("is_patient", False),
            )
        elif msg_type == "remove_member":
            pid = msg.get("person_id", "")
            if pid:
                with self._lock:
                    self._members.pop(pid, None)
                self._log(f"Face WS: đã xóa thành viên {pid}")
