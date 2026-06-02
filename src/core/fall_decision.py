"""
src/core/fall_decision.py — Fall decision state machine, tách ra khỏi UI layer.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

from schemas import PoseState

_UPRIGHT_STATES = {
    str(PoseState.STANDING),
    str(PoseState.WALKING),
    str(PoseState.SITTING),
}


@dataclass
class FallDecision:
    final_fall: bool
    ai_sees_fall: bool
    rule_fall: bool
    source: str          # "Rules + AI" | "AI Transformer" | "Rule-based" | "None"
    lying_duration: float
    current_lying: bool
    was_upright: bool


class FallDecisionEngine:
    """
    Giữ state machine (_lying_start_t / _non_lying_streak) và tính
    quyết định té ngã cuối cùng từ rule-based + Transformer AI.
    """

    def __init__(self):
        self._lying_start_t    = 0.0
        self._non_lying_streak = 0

    def update(self, result, tr, vel_y_abs: float) -> FallDecision:
        """
        result  — DetectionResult từ fall_detector
        tr      — TransformerResult | None từ transformer_engine
        vel_y_abs — abs(result.velocity_y)
        """
        rule_fall     = result.is_falling
        current_lying = str(result.state) == str(PoseState.LYING)
        was_upright   = str(result.prev_state) in _UPRIGHT_STATES

        # Chống flicker: chỉ reset _lying_start_t sau ≥ 15 frame không nằm liên tiếp
        if current_lying:
            self._non_lying_streak = 0
            if self._lying_start_t == 0.0:
                self._lying_start_t = time.time()
        else:
            self._non_lying_streak += 1
            if self._non_lying_streak >= 15:
                self._lying_start_t = 0.0

        lying_duration = (
            (time.time() - self._lying_start_t)
            if (current_lying and self._lying_start_t > 0)
            else float("inf")
        )

        ai_sees_fall = (
            tr is not None
            and tr.ready
            and tr.is_fall
            and vel_y_abs > 40.0
            and current_lying
            and was_upright
            and lying_duration < 1.0
        )

        final_fall = ai_sees_fall and rule_fall

        if rule_fall and final_fall:
            source = "Rules + AI"
        elif final_fall:
            source = "AI Transformer"
        elif rule_fall:
            source = "Rule-based"
        else:
            source = "None"

        return FallDecision(
            final_fall    = final_fall,
            ai_sees_fall  = ai_sees_fall,
            rule_fall     = rule_fall,
            source        = source,
            lying_duration= lying_duration,
            current_lying = current_lying,
            was_upright   = was_upright,
        )

    def reset(self):
        self._lying_start_t    = 0.0
        self._non_lying_streak = 0
