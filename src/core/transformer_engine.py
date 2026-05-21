"""
src/core/transformer_engine.py
Sliding-window Transformer inference for fall / non-fall classification.

Model input : (1, num_frames=30, input_dim=200)
              MediaPipe 33 kp × 4 (x,y,z,vis) || YOLO 17 kp × 4 (nx,ny,0,conf)
Model output: (1, 2) → softmax → [p_non_fall, p_fall]

Lưu ý:
  - KHÔNG dùng StandardScaler — extract_keypoints đã normalize per-sequence
  - Inference normalize per-sequence giống hệt extract_keypoints
  - num_frames đọc từ checkpoint config (không hardcode)
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from dataclasses import dataclass
from typing import Optional

_CKPT_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "train_tranformer", "dataset", "checkpoints1",
    )
)

# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class TransformerResult:
    is_fall:    bool  = False
    confidence: float = 0.0   # P(fall) từ softmax
    ready:      bool  = False  # True khi buffer đã đủ num_frames


# ── Model (giống hệt training) ────────────────────────────────────────────────

class _PoseTransformer(nn.Module):
    def __init__(self, input_dim: int, num_frames: int,
                 d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.4,
                 num_classes: int = 2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.pos_emb   = nn.Parameter(
            torch.randn(1, num_frames + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B   = x.size(0)
        x   = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_emb
        x   = self.transformer(x)
        return self.head(x[:, 0])


# ── Engine ────────────────────────────────────────────────────────────────────

class TransformerEngine:
    """
    Real-time fall classifier dùng sliding window của pose keypoints.

    Pipeline:
        raw_kp = pose_engine.process(frame)   # ndarray shape (200,)
        transformer_engine.push(raw_kp)
        result = transformer_engine.get()      # TransformerResult

    Normalize:
        Dùng per-sequence normalization giống extract_keypoints.py
        KHÔNG dùng StandardScaler
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        fall_threshold: float = 0.6,   # P(fall) để kết luận là fall
    ):
        self._ckpt_dir = checkpoint_dir or _CKPT_DIR
        self._fall_thr = fall_threshold

        self._device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model:      Optional[_PoseTransformer] = None
        self._num_frames: int                        = 30   # sẽ đọc từ checkpoint
        self._input_dim:  int                        = 200
        self._buffer:     deque                      = deque(maxlen=30)
        self._frame_count: int                       = 0
        self._last:       TransformerResult          = TransformerResult()
        self._loaded:     bool                       = False

        # predict_every: infer mỗi N frames — tính sau khi load checkpoint
        self._predict_every: int = 8

        self._load()

    # ── Load checkpoint ───────────────────────────────────────────────────────

    def _load(self) -> None:
        ckpt_path = os.path.join(self._ckpt_dir, "best_model.pth")

        if not os.path.exists(ckpt_path):
            print(f"[TransformerEngine] Checkpoint không tìm thấy: {ckpt_path}")
            return

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg  = ckpt["config"]

            # Đọc config từ checkpoint — không hardcode
            self._num_frames = int(cfg["num_frames"])
            self._input_dim  = int(cfg["input_dim"])

            # Kiểm tra data đã normalized trong extract chưa
            if not cfg.get("normalized", False):
                print("[TransformerEngine] WARNING: checkpoint không có "
                      "normalized=True — kiểm tra lại extract_keypoints.py")

            # Buffer size = num_frames
            self._buffer = deque(maxlen=self._num_frames)

            # predict_every = 25% buffer (infer 4 lần mỗi window đầy)
            self._predict_every = max(1, self._num_frames // 4)

            # Load model
            self._model = _PoseTransformer(
                input_dim  = self._input_dim,
                num_frames = self._num_frames,
                d_model    = int(cfg.get("d_model",    128)),
                nhead      = int(cfg.get("nhead",        4)),
                num_layers = int(cfg.get("num_layers",   2)),
                dropout    = float(cfg.get("dropout",  0.4)),
            )
            self._model.load_state_dict(ckpt["model_state"])
            self._model.eval()
            self._model = self._model.to(self._device)
            self._loaded = True

            val_acc  = float(ckpt.get("val_acc", 0))
            mean_acc = float(ckpt.get("kfold_mean", 0))
            print(
                f"[TransformerEngine] Loaded OK\n"
                f"  device        = {self._device}\n"
                f"  num_frames    = {self._num_frames}\n"
                f"  input_dim     = {self._input_dim}\n"
                f"  predict_every = {self._predict_every} frames\n"
                f"  fall_threshold= {self._fall_thr}\n"
                f"  val_acc       = {val_acc:.1f}%  "
                f"kfold_mean = {mean_acc:.1f}%\n"
                f"  normalize     = per-sequence (không dùng scaler)"
            )

        except Exception as exc:
            print(f"[TransformerEngine] Load thất bại: {exc}")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def buffer_ready(self) -> bool:
        """True khi buffer đã đủ num_frames"""
        return len(self._buffer) == self._num_frames

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, raw_kp: np.ndarray) -> None:
        """
        Đẩy 1 keypoint vector (200,) vào sliding window.
        Tự động infer mỗi predict_every frames khi buffer đầy.

        Điều kiện infer:
          1. Buffer đủ num_frames
          2. frame_count chia hết cho predict_every
          3. >= 50% frames trong buffer có pose detect được
        """
        if not self._loaded:
            return

        self._buffer.append(raw_kp.astype(np.float32))
        self._frame_count += 1

        if not self.buffer_ready:
            return

        if self._frame_count % self._predict_every != 0:
            return

        # Kiểm tra >= 50% frames có pose (MediaPipe part non-zero)
        buf_arr   = np.array(self._buffer)              # (T, 200)
        mp_part   = buf_arr[:, :132]                    # MediaPipe 33kp×4
        has_pose  = np.any(np.abs(mp_part) > 1e-3, axis=1)  # (T,) bool
        pose_ratio = has_pose.mean()

        if pose_ratio < 0.5:
            # Quá ít frame detect được người → không infer
            return

        self._last = self._infer(buf_arr)

    def get(self) -> TransformerResult:
        """Trả về kết quả infer gần nhất (cached)."""
        return TransformerResult(
            is_fall    = self._last.is_fall,
            confidence = self._last.confidence,
            ready      = self.buffer_ready,
        )

    def reset(self) -> None:
        """Reset buffer — gọi sau khi xử lý xong 1 người hoặc scene thay đổi."""
        self._buffer.clear()
        self._frame_count = 0
        self._last = TransformerResult()

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer(self, buf_arr: np.ndarray) -> TransformerResult:
        """
        Normalize per-sequence (giống extract_keypoints.py) rồi infer.
        KHÔNG dùng StandardScaler.
        """
        X = buf_arr.copy()  # (T, 200)

        # Per-sequence normalization — giống normalize_sequence() trong extract
        mean = X.mean(axis=0, keepdims=True)        # (1, 200)
        std  = X.std(axis=0, keepdims=True)         # (1, 200)
        std  = np.where(std < 1e-6, 1.0, std)      # tránh chia 0
        X    = (X - mean) / std

        t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self._device)  # (1, T, 200)

        with torch.no_grad():
            logits = self._model(t)
            probs  = torch.softmax(logits, dim=1)[0]
            pred   = int(logits.argmax(1).item())

        fall_conf = float(probs[1])
        is_fall   = (pred == 1 and fall_conf >= self._fall_thr)

        return TransformerResult(
            is_fall    = is_fall,
            confidence = fall_conf,
            ready      = True,
        )