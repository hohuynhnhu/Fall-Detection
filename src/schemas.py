"""
schemas.py — Pydantic schemas cho Fall Detection Desktop
Tự chứa hoàn toàn, không phụ thuộc external package ngoài pydantic.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

__all__ = [
    "PoseState", "EventType",
    "ThresholdConfig", "FeatureConfig",
    "BodyMetricsPayload", "FallEvent", "FallEventResponse",
    "PoseEvent", "PersonDetectedPayload", "PatientPoseEvent",
    "HeartbeatPayload",
    "FamilyMember", "AddFamilyMemberPayload", "FamilyMembersResponse",
    "FamilyMemberWithImage", "FamilyMembersAllResponse",
    "NewMemberWSMessage",
]


# ── Enums ─────────────────────────────────────────────────────────────────────

class PoseState(str, Enum):
    STANDING = "STANDING"
    SITTING  = "SITTING"
    LYING    = "LYING"
    FALLING  = "FALLING"
    WALKING  = "WALKING"
    UNKNOWN  = "UNKNOWN"


class EventType(str, Enum):
    FALL             = "fall"
    POSE_CHANGE      = "pose_change"
    PERSON_DETECTED  = "person_detected"
    PATIENT_POSE     = "patient_pose"


# ── Config ────────────────────────────────────────────────────────────────────

class ThresholdConfig(BaseModel):
    """Ngưỡng phân loại tư thế và phát hiện té ngã — đồng bộ từ backend về."""

    # Pose classification
    min_confidence:          float = Field(0.5,  description="Confidence tối thiểu")
    body_angle_lying:        float = Field(50.0, description="Góc thân (°) để coi là nằm")
    aspect_ratio_lying:      float = Field(0.55, description="H/W bbox để coi là nằm")

    # Bending guard
    bend_shoulder_drop_threshold: float = 40.0
    bend_hip_max_drop:            float = 25.0

    # Fall detection
    fall_velocity_threshold: float = Field(40.0, description="Vận tốc xuống (px/s) trigger fall")
    fall_confirm_frames:     int   = Field(10,   description="Số frame LYING liên tiếp để confirm")
    fall_history_window:     int   = Field(30,   description="Cửa sổ frame tính velocity")
    fall_transition_max_s:   float = Field(2.5,  description="Thời gian tối đa đứng→nằm để coi là té")

    # Walking
    walk_velocity_threshold:  float = Field(20.0, description="Vận tốc ngang (px/s) detect đi")
    walk_knee_lift_threshold: float = Field(0.08, description="Ngưỡng nâng gối chuẩn hóa")
    walk_alternating_window:  int   = Field(15,   description="Số frame kiểm tra gối xen kẽ")

    # Sleep-as-fall
    sleep_confirm_frames: int = Field(150, description="Số frame LYING để coi là ngủ = té")

    sitting_ar_min: float = Field(0.50, description="aspect_ratio min khi ngồi")
    sitting_ar_max: float = Field(1.70, description="aspect_ratio max khi ngồi")
    sitting_angle_max: float = Field(40.0, description="body_angle max khi ngồi")
    sitting_hip_ankle_max: float = Field(0.75, description="hip_ankle_ratio max khi ngồi")

    # Camera
    camera_index:     int  = Field(0)
    flip_horizontal:  bool = Field(True)
    model_complexity: int  = Field(1, ge=0, le=2)

    class Config:
        from_attributes = True

class FeatureConfig(BaseModel):
    """Feature flags controlled by backend (mobile app) — not local UI state."""
    enable_face_recognition:        bool  = Field(True,  description="Bật/tắt nhận diện khuôn mặt + đăng ký từ mobile")
    enable_patient_pose_notification: bool = Field(True,  description="Bật/tắt gửi thông báo tư thế bệnh nhân về mobile")
    enable_sound_detection:         bool  = Field(True,  description="Bật/tắt phát hiện âm thanh (YAMNet)")
    sleep_as_fall:                  bool  = Field(False, description="Nằm lâu = té ngã")
    sound_listen_seconds:           float = Field(3.0, ge=1.0, le=10.0,
                                                  description="Thời gian lắng nghe âm thanh sau khi phát hiện té")

    class Config:
        from_attributes = True


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
    camera_id:    str   = "cam_0"
    timestamp:    float = 0.0
    confidence:   float = Field(0.0, ge=0, le=1)
    person_count: int   = 1


# ── Family Members ────────────────────────────────────────────────────────────

class FamilyMember(BaseModel):
    person_id:    str
    name:         str
    role:         str  = "family"
    sample_count: int  = 0
    is_patient:   bool = False


class AddFamilyMemberPayload(BaseModel):
    """POST /family-members/register"""
    person_id:      str
    name:           str
    role:           str  = "family"
    is_patient:     bool = False
    face_image_url: str  = ""   # URL Cloudinary; rỗng khi đăng ký từ desktop


class FamilyMembersResponse(BaseModel):
    """GET /family-members"""
    members: List[FamilyMember] = []


class FamilyMemberWithImage(BaseModel):
    """GET /family-members/all — kèm URL ảnh khuôn mặt để desktop download + encode."""
    person_id:      str
    name:           str
    role:           str  = "family"
    is_patient:     bool = False
    face_image_url: str  = ""   # URL backend lưu ảnh gốc từ mobile


class FamilyMembersAllResponse(BaseModel):
    """GET /family-members/all"""
    members: List[FamilyMemberWithImage] = []


# ── Heartbeat ─────────────────────────────────────────────────────────────────

class HeartbeatPayload(BaseModel):
    """POST /events/heartbeat"""
    camera_id: str       = "cam_0"
    timestamp: float     = 0.0
    fps:       float     = Field(0.0, ge=0)
    state:     PoseState = PoseState.UNKNOWN


# ── Patient Pose Event ────────────────────────────────────────────────────────

class PatientPoseEvent(BaseModel):
    """POST /events/patient-pose — Thông báo tư thế bệnh nhân theo thời gian thực."""
    event_type:   EventType              = EventType.PATIENT_POSE
    camera_id:    str                    = "cam_0"
    timestamp:    float                  = 0.0
    person_id:    str                    = ""
    person_name:  str                    = ""
    state:        PoseState              = PoseState.UNKNOWN
    prev_state:   PoseState              = PoseState.UNKNOWN
    metrics:      Optional[BodyMetricsPayload] = None
    frame_id:     int                    = 0


# ── WebSocket push từ backend ─────────────────────────────────────────────────

class NewMemberWSMessage(BaseModel):
    """Backend push qua WebSocket /ws/desktop khi mobile đăng ký thành viên mới."""
    type:               str  = "new_member"
    person_id:          str  = ""
    name:               str  = ""
    role:               str  = "family"
    is_patient:         bool = False
    face_image_base64:  str  = ""   # JPEG bytes encode base64


# ── Face Recognition Log ──────────────────────────────────────────────────────

class FaceLogPayload(BaseModel):
    """POST /face-logs — gửi sau mỗi lần nhận diện khuôn mặt thành công."""
    person_id:     str
    name:          str
    is_patient:    bool  = False
    confidence:    float = Field(0.0, ge=0.0, le=1.0)
    camera_id:     str   = "cam_0"
    recognized_at: float = 0.0   # unix timestamp; 0 → backend dùng server time


# ── Legacy aliases (backward compat) ─────────────────────────────────────────

FallEventPayload = FallEvent
