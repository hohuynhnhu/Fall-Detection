"""
src/core/face_recognizer_if.py — InsightFace buffalo_l (ArcFace) wrapper.

Downloads buffalo_l (~300 MB) on first run.
Prefers CUDAExecutionProvider, falls back to CPUExecutionProvider.
"""
from __future__ import annotations
import os
import sys
import numpy as np
from typing import List, Optional

# ── Expose CUDA / cuDNN DLLs cho onnxruntime-gpu (Windows) ──────────────────
# onnxruntime-gpu 1.17+ bundle CUDA runtime nhưng KHÔNG bundle cuDNN.
# PyTorch có đủ cudnn*_9.dll trong torch/lib/.
# onnxruntime-gpu có onnxruntime_providers_cuda.dll trong onnxruntime/capi/.
# Cần add cả 2 thư mục + pre-load cuDNN để _pybind_state.pyd tìm thấy DLLs.
_dll_dirs: list = []   # giữ reference → tránh GC xoá DLL path
if sys.platform == "win32":
    try:
        import ctypes
        import importlib
        import torch as _torch

        # 1. torch/lib — cuDNN 9 DLLs
        _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            _dll_dirs.append(os.add_dll_directory(_torch_lib))
            for _dll_name in [
                "cudnn64_9.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll",
                "cudnn_adv64_9.dll", "cudnn_graph64_9.dll",
                "cudnn_engines_precompiled64_9.dll", "cudnn_heuristic64_9.dll",
            ]:
                _p = os.path.join(_torch_lib, _dll_name)
                if os.path.isfile(_p):
                    try:
                        ctypes.WinDLL(_p)
                    except Exception:
                        pass

        # 2. onnxruntime/capi — onnxruntime.dll, providers_cuda.dll, cudart DLLs
        _ort_spec = importlib.util.find_spec("onnxruntime")
        if _ort_spec and _ort_spec.submodule_search_locations:
            for _ort_root in _ort_spec.submodule_search_locations:
                for _sub in ("", "capi"):
                    _d = os.path.join(_ort_root, _sub) if _sub else _ort_root
                    if os.path.isdir(_d):
                        try:
                            _dll_dirs.append(os.add_dll_directory(_d))
                        except Exception:
                            pass
    except Exception:
        pass

try:
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_OK = True
except ImportError:
    _INSIGHTFACE_OK = False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (not required to be pre-normalised)."""
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-8 or bn < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


class FaceRecognizer:
    """InsightFace buffalo_l — returns L2-normalised 512-d ArcFace embeddings."""

    def __init__(self, det_size: tuple = (320, 320)):
        if not _INSIGHTFACE_OK:
            raise ImportError("pip install insightface onnxruntime")
        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=det_size)

    def get_embedding_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Return 512-d normalised embedding of largest detected face, or None."""
        faces = self._app.get(image)
        if not faces:
            return None
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        return face.normed_embedding.copy()

    def get_faces_in_crop(self, crop: np.ndarray) -> List[dict]:
        """
        Detect faces within a person crop.
        Returns list of {bbox: [x1,y1,x2,y2], embedding: ndarray(512), det_score: float}.
        """
        faces = self._app.get(crop)
        return [
            {
                "bbox":      face.bbox.astype(int).tolist(),
                "embedding": face.normed_embedding.copy(),
                "det_score": float(face.det_score),
            }
            for face in faces
        ]
