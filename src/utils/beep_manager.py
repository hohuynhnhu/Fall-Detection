"""
src/utils/beep_manager.py — Beep cooldown tách ra khỏi UI layer.
"""
from __future__ import annotations
import time
import threading


def _beep_fall():
    def _play():
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1000, 200)
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()


class BeepManager:
    def __init__(self, cooldown_s: float = 3.0):
        self._cooldown_s = cooldown_s
        self._last_t     = 0.0

    def trigger(self, should_beep: bool) -> bool:
        """Phát beep nếu should_beep=True và cooldown đã qua. Trả True nếu beep phát."""
        if not should_beep:
            return False
        now = time.time()
        if now - self._last_t >= self._cooldown_s:
            _beep_fall()
            self._last_t = now
            return True
        return False
