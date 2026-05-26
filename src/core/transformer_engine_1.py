"""
src/core/transformer_engine_1.py
Sliding-window Transformer inference — dùng config num_frames=30, normalized=True.

Config:
  num_frames : 30  (window 3s @ 10fps hoặc 1s @ 30fps)
  input_dim  : 200
  normalized : True  ← data đã normalize khi extract, inference normalize per-sequence
  fusion     : True  (MediaPipe 33kp×4 + YOLO 17kp×4 = 200)
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from dataclasses import dataclass
from typing import Optional

# ── Đường dẫn checkpoint — chỉnh lại nếu cần ─────────────────────────────────
_CKPT_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "train_tranformer", "dataset_1", "checkpoints1",
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

class TransformerEngine1:
    """
    Transformer engine dùng model num_frames=30.

    Khác với TransformerEngine (num_frames=64):
    - Buffer nhỏ hơn → phản hồi nhanh hơn (~1s @ 30fps)
    - normalized=True trong config → inference normalize per-sequence

    Dùng để test xem model 30-frame hoạt động tốt hơn không.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        fall_threshold:  float = 0.6,
    ):
        # Cho phép truyền path file .pth trực tiếp hoặc dùng default
        if checkpoint_path and checkpoint_path.endswith(".pth"):
            self._ckpt_path = checkpoint_path
        else:
            ckpt_dir = checkpoint_path or _CKPT_DIR
            self._ckpt_path = os.path.join(ckpt_dir, "best_model.pth")

        self._fall_thr    = fall_threshold
        self._device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model:      Optional[_PoseTransformer] = None
        self._num_frames: int   = 30   # hardcode theo config
        self._input_dim:  int   = 200
        self._buffer:     deque = deque(maxlen=30)
        self._frame_count: int  = 0
        self._last:       TransformerResult = TransformerResult()
        self._loaded:     bool  = False
        self._predict_every: int = 8   # infer mỗi 8 frame (~ 4 lần/window)

        self._load()

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._ckpt_path):
            print(f"[TransformerEngine1] Checkpoint không tìm thấy: {self._ckpt_path}")
            return

        try:
            ckpt = torch.load(self._ckpt_path, map_location="cpu", weights_only=False)
            cfg  = ckpt.get("config", {})

            # Đọc num_frames từ checkpoint — ưu tiên checkpoint, fallback về 30
            ckpt_frames = int(cfg.get("num_frames", 30))
            if ckpt_frames != self._num_frames:
                print(
                    f"[TransformerEngine1] WARNING: checkpoint num_frames={ckpt_frames} "
                    f"khác config num_frames={self._num_frames} — dùng {ckpt_frames}"
                )
                self._num_frames = ckpt_frames
                self._buffer     = deque(maxlen=self._num_frames)

            self._input_dim     = int(cfg.get("input_dim", 200))
            self._predict_every = max(1, self._num_frames // 4)

            # Kiểm tra normalized
            normalized = cfg.get("normalized", False)
            if not normalized:
                print("[TransformerEngine1] WARNING: checkpoint không có normalized=True")
            else:
                print("[TransformerEngine1] normalized=True ✓")

            # Build model
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
            self._model  = self._model.to(self._device)
            self._loaded = True

            val_acc  = float(ckpt.get("val_acc",    0))
            mean_acc = float(ckpt.get("kfold_mean", 0))
            print(
                f"[TransformerEngine1] Loaded OK\n"
                f"  device        = {self._device}\n"
                f"  num_frames    = {self._num_frames}\n"
                f"  input_dim     = {self._input_dim}\n"
                f"  predict_every = {self._predict_every} frames\n"
                f"  fall_threshold= {self._fall_thr}\n"
                f"  val_acc       = {val_acc:.1f}%  kfold_mean = {mean_acc:.1f}%\n"
                f"  normalize     = per-sequence"
            )

        except Exception as exc:
            print(f"[TransformerEngine1] Load thất bại: {exc}")

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
        return len(self._buffer) == self._num_frames

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, raw_kp: np.ndarray) -> None:
        """Đẩy 1 keypoint vector (200,) vào buffer, tự infer khi đủ."""
        if not self._loaded:
            return

        self._buffer.append(raw_kp.astype(np.float32))
        self._frame_count += 1

        if not self.buffer_ready:
            return

        if self._frame_count % self._predict_every != 0:
            return

        buf_arr    = np.array(self._buffer)       # (T, 200)
        mp_part    = buf_arr[:, :132]             # MediaPipe 33kp×4
        has_pose   = np.any(np.abs(mp_part) > 1e-3, axis=1)
        pose_ratio = has_pose.mean()

        if pose_ratio < 0.5:
            return  # quá ít frame có pose → bỏ qua

        self._last = self._infer(buf_arr)

    def get(self) -> TransformerResult:
        return TransformerResult(
            is_fall    = self._last.is_fall,
            confidence = self._last.confidence,
            ready      = self.buffer_ready,
        )

    def reset(self) -> None:
        self._buffer.clear()
        self._frame_count = 0
        self._last = TransformerResult()

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer(self, buf_arr: np.ndarray) -> TransformerResult:
        """Per-sequence normalize rồi infer."""
        X    = buf_arr.copy()
        mean = X.mean(axis=0, keepdims=True)
        std  = X.std(axis=0,  keepdims=True)
        std  = np.where(std < 1e-6, 1.0, std)
        X    = (X - mean) / std

        t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self._device)

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