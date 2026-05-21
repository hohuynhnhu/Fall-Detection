"""
src/core/audio_engine.py — YAMNet-based fall-sound detection

Loads YAMNet in a background thread (non-blocking startup).
Call trigger() after fall detection; the audio result arrives via on_result callback
3-10 seconds later (configurable listen window).

Optional deps: tensorflow, tensorflow-hub, sounddevice
If unavailable, status = "unavailable" and trigger() is a no-op.
"""
from __future__ import annotations
import threading
import time
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional

# AudioSet class keywords associated with fall-related sounds
_FALL_KEYWORDS: List[str] = [
    "thud", "impact", "crash", "bang", "smash", "thwack", "slam",
    "scream", "screaming", "shout", "shouting",
    "cry", "crying", "sob", "wail", "whimper", "yelp",
    "grunt", "groan", "moan",
    "glass", "breaking", "shatter",
    "hit", "knock",
]


@dataclass
class AudioResult:
    detected:    bool  = False   # True if a fall-related sound was found
    sound_class: str   = ""      # Top matching AudioSet class name
    confidence:  float = 0.0     # Score in [0, 1]
    timestamp:   float = 0.0


class AudioEngine:
    """
    YAMNet wrapper triggered on fall detection.

    Usage:
        engine = AudioEngine(listen_seconds=3.0, on_result=my_callback)
        # ... when fall detected:
        engine.trigger()
        # my_callback(AudioResult(...)) called in a daemon thread ~3s later
    """

    def __init__(
        self,
        listen_seconds: float = 3.0,
        sample_rate:    int   = 16000,
        on_result: Optional[Callable[[AudioResult], None]] = None,
    ):
        self.listen_seconds = listen_seconds
        self.sample_rate    = sample_rate
        self.on_result      = on_result
        self.enabled        = True

        self.loaded  = False
        self.status  = "loading"   # loading | ready | unavailable

        self._model:       object           = None
        self._class_names: Optional[List[str]] = None
        self._busy         = False

        threading.Thread(
            target=self._load, daemon=True, name="AudioEngine-load"
        ).start()

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            import tensorflow_hub as hub  # type: ignore
            self._model       = hub.load("https://tfhub.dev/google/yamnet/1")
            self._class_names = self._fetch_class_names()
            self.loaded = True
            self.status = "ready"
        except Exception as exc:
            self.status = f"unavailable ({exc})"

    @staticmethod
    def _fetch_class_names() -> List[str]:
        import csv
        import io
        import urllib.request
        url = (
            "https://raw.githubusercontent.com/tensorflow/models/"
            "master/research/audioset/yamnet/yamnet_class_map.csv"
        )
        resp   = urllib.request.urlopen(url, timeout=10)
        reader = csv.DictReader(io.TextIOWrapper(resp))
        return [row["display_name"] for row in reader]

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def is_busy(self) -> bool:
        return self._busy

    def trigger(self):
        """Non-blocking: spawn a listener thread if not already listening."""
        if not self.loaded or not self.enabled or self._busy:
            return
        threading.Thread(
            target=self._run, daemon=True, name="AudioEngine-listen"
        ).start()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run(self):
        self._busy = True
        try:
            result = self._record_and_classify()
            if self.on_result:
                self.on_result(result)
        finally:
            self._busy = False

    def _record_and_classify(self) -> AudioResult:
        try:
            import sounddevice as sd  # type: ignore

            audio: np.ndarray = sd.rec(
                int(self.listen_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            waveform = audio.flatten()

            scores, _, _ = self._model(waveform)
            mean_scores  = scores.numpy().mean(axis=0)   # (521,)

            # Check top-5 classes for any fall-related keyword
            top5 = np.argsort(mean_scores)[::-1][:5]
            for idx in top5:
                name = self._class_names[idx] if self._class_names else str(idx)
                conf = float(mean_scores[idx])
                if any(kw in name.lower() for kw in _FALL_KEYWORDS):
                    return AudioResult(
                        detected=True, sound_class=name,
                        confidence=conf, timestamp=time.time(),
                    )

            top_idx  = int(np.argmax(mean_scores))
            top_name = (self._class_names[top_idx]
                        if self._class_names else str(top_idx))
            return AudioResult(
                detected=False,
                sound_class=top_name,
                confidence=float(mean_scores[top_idx]),
                timestamp=time.time(),
            )

        except Exception as exc:
            return AudioResult(
                detected=False, sound_class=f"err: {exc}",
                confidence=0.0, timestamp=time.time(),
            )
