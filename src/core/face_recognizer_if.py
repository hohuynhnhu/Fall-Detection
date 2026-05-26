"""
src/core/face_recognizer_if.py — InsightFace buffalo_l (ArcFace) wrapper.

Downloads buffalo_l (~300 MB) on first run.
Prefers CUDAExecutionProvider, falls back to CPUExecutionProvider.
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional

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
