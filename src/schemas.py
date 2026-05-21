"""
schemas.py — Pydantic schemas cho Fall Detection Desktop
Tự chứa hoàn toàn, không phụ thuộc external package ngoài pydantic.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

__all__ = [
    "PoseState", "DeviceStatusEnum", "EventType",
    "ThresholdConfig", "FeatureConfig", "ConfigResponse",
    "BodyMetricsPayload", "FallEvent", "FallEventResponse",
    "PoseEvent", "PersonDetectedPayload",
    "FamilyMember", "AddFamilyMemberPayload", "FamilyMembersResponse",
    "DeviceStatusPayload", "DeviceStatusResponse",
]


# ── Enums ─────────────────────────────────────────────────────────────────────

class PoseState(str, Enum):
    STANDING = "STANDING"
    SITTING  = "SITTING"
    LYING    = "LYING"
    FALLING  = "FALLING"
    WALKING  = "WALKING"
    UNKNOWN  = "UNKNOWN"


class DeviceStatusEnum(str, Enum):
    ONLINE   = "online"
    IDLE     = "idle"
    ERROR    = "error"
    OFFLINE  = "offline"


class EventType(str, Enum):
    FALL             = "fall"
    POSE_CHANGE      = "pose_change"
    PERSON_DETECTED  = "person_detected"


# ── Config ────────────────────────────────────────────────────────────────────

class ThresholdConfig(BaseModel):
    """Ngưỡng phân loại tư thế và phát hiện té ngã — đồng bộ từ backend về."""
    # Pose classification
    body_angle_lying:        float = Field(65.0,  description="Góc cột sống (°) để coi là nằm")
    body_angle_sitting:      float = Field(45.0,  description="Góc cột sống (°) để coi là ngồi")
    aspect_ratio_lying:      float = Field(0.55,  description="H/W bbox < ngưỡng → nằm")
    torso_ratio_sitting:     float = Field(0.42,  description="Tỉ lệ torso để phân biệt đứng/ngồi")
    bend_shoulder_drop_threshold: float = 40.0  # px shoulder xuống so với baseline
    bend_hip_max_drop: float = 25.0  # px hip tối đa xuống (nếu lớn hơn → fall)
    # Fall detection
    fall_velocity_threshold: float = Field(80.0,  description="Vận tốc xuống (px/s) trigger fall")
    fall_confirm_frames:     int   = Field(5,     description="Số frame LYING liên tiếp để confirm")
    fall_history_window:     int   = Field(30,    description="Cửa sổ frame tính velocity")
    # Walking
    walk_velocity_threshold:  float = Field(20.0,  description="Vận tốc ngang (px/s) detect đi")
    walk_knee_lift_threshold: float = Field(0.08,  description="Ngưỡng nâng gối chuẩn hóa")
    walk_alternating_window:  int   = Field(15,    description="Số frame kiểm tra gối xen kẽ")
    # Fall transition time guard
    fall_transition_max_s:   float = Field(2.0,   description="Nếu từ đứng → nằm mất hơn N giây → không phải té")
    # Sleep-as-fall: treat prolonged lying as fall (feature-flag controlled)
    sleep_confirm_frames:    int   = Field(150, description="Số frame LYING để coi là ngủ/nằm = té (≈5s @ 30fps)")
    # Camera
    camera_index:            int   = Field(0)
    flip_horizontal:         bool  = Field(True)
    model_complexity:        int   = Field(1, ge=0, le=2)

    class Config:
        from_attributes = True


class FeatureConfig(BaseModel):
    """Feature flags controlled by backend (mobile app) — not local UI state."""
    enable_face_recognition: bool  = Field(True,  description="Bật/tắt nhận diện khuôn mặt")
    enable_sound_detection:  bool  = Field(True,  description="Bật/tắt phát hiện âm thanh (YAMNet)")
    sleep_as_fall:           bool  = Field(False, description="Nằm lâu = té ngã")
    sound_listen_seconds:    float = Field(3.0, ge=1.0, le=10.0,
                                           description="Thời gian lắng nghe âm thanh sau khi phát hiện té")

    class Config:
        from_attributes = True


class ConfigResponse(BaseModel):
    """Response body cho GET /config."""
    device_id:  str             = "cam_0"
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    features:   FeatureConfig   = Field(default_factory=FeatureConfig)


# ── Body Metrics ──────────────────────────────────────────────────────────────

class BodyMetricsPayload(BaseModel):
    body_angle:   float = 0.0
    aspect_ratio: float = 1.5
    center_x:     float = 0.0
    center_y:     float = 0.0
    confidence:   float = 0.0
    hip_y:        float = 0.0
    shoulder_y:   float = 0.0
    ankle_y:      float = 0.0


# ── Fall Event ────────────────────────────────────────────────────────────────

class FallEvent(BaseModel):
    """POST /events/fall"""
    event_type:        EventType     = EventType.FALL
    camera_id:         str           = "cam_0"
    timestamp:         float         = 0.0
    state:             PoseState     = PoseState.FALLING
    state_before:      PoseState     = PoseState.UNKNOWN
    velocity_px_per_s: float         = 0.0
    max_velocity:      float         = 0.0
    body_angle:        float         = 0.0
    confidence:        float         = Field(0.0, ge=0, le=1)
    frame_id:          int           = 0
    clip_url:          Optional[str] = None


class FallEventResponse(BaseModel):
    event_id:  str
    received:  bool  = True
    timestamp: float = 0.0


# ── Pose Event ────────────────────────────────────────────────────────────────

class PoseEvent(BaseModel):
    """POST /events/pose"""
    event_type:        EventType              = EventType.POSE_CHANGE
    camera_id:         str                    = "cam_0"
    timestamp:         float                  = 0.0
    state:             PoseState              = PoseState.UNKNOWN
    prev_state:        PoseState              = PoseState.UNKNOWN
    velocity_px_per_s: float                  = 0.0
    metrics:           Optional[BodyMetricsPayload] = None
    frame_id:          int                    = 0


# ── Person Detected Event ─────────────────────────────────────────────────────

class PersonDetectedPayload(BaseModel):
    """POST /events/person-detected"""
    event_type:   EventType = EventType.PERSON_DETECTED
    camera_id:    str       = "cam_0"
    timestamp:    float     = 0.0
    person_id:    str       = "unknown"
    person_name:  str       = ""
    confidence:   float     = Field(0.0, ge=0, le=1)
    frame_id:     int       = 0


# ── Family Members ────────────────────────────────────────────────────────────

class FamilyMember(BaseModel):
    person_id:    str
    name:         str
    role:         str = "family"
    sample_count: int = 0


class AddFamilyMemberPayload(BaseModel):
    """POST /family-members"""
    person_id: str
    name:      str
    role:      str = "family"


class FamilyMembersResponse(BaseModel):
    """GET /family-members"""
    members: List[FamilyMember] = []


# ── Device Status ─────────────────────────────────────────────────────────────

class DeviceStatusPayload(BaseModel):
    """POST /device/status"""
    device_id:      str              = "cam_0"
    timestamp:      float            = 0.0
    status:         DeviceStatusEnum = DeviceStatusEnum.ONLINE
    fps:            float            = Field(0.0, ge=0)
    current_pose:   PoseState        = PoseState.UNKNOWN
    uptime_seconds: float            = Field(0.0, ge=0)
    error_message:  Optional[str]    = None


class DeviceStatusResponse(BaseModel):
    received:  bool  = True
    timestamp: float = 0.0


# ── Legacy aliases (backward compat) ─────────────────────────────────────────

FallEventPayload = FallEvent
