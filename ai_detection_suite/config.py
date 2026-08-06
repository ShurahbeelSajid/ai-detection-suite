"""Validated configuration and presets for the command-line entry points."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_MODEL_ROOT = PACKAGE_ROOT / "models"


def default_model_root() -> Path:
    """Return the explicit model trust root, or the bundled models directory."""

    configured = os.environ.get("AI_DETECTION_MODEL_DIR")
    return Path(configured).expanduser() if configured else DEFAULT_MODEL_ROOT


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    camera_index: int = 0
    capture_width: int = 960
    capture_height: int = 540
    capture_fps: int = 30
    processing_scale: float = 0.65
    face_stride: int = 2
    hand_stride: int = 1
    hand_ttl: float = 0.50
    object_hz: float = 10.0
    attribute_hz: float = 0.75
    object_ttl: float = 0.75
    attribute_ttl: float = 2.5
    object_model: str = "yolo11n.pt"
    model_root: Path = dataclass_field(default_factory=default_model_root)
    object_img_size: int = 384
    object_confidence: float = 0.40
    object_iou: float = 0.55
    object_max_detections: int = 20
    device: str = "auto"
    hand_backend: str = "auto"
    hand_model: Path | None = None
    hand_model_sha256: str | None = None
    enable_faces: bool = True
    enable_hands: bool = True
    enable_objects: bool = True
    enable_face_attributes: bool = False
    allow_unverified_model: bool = False
    allow_model_downloads: bool = False
    strict_features: bool = False
    mirror: bool = True
    max_frames: int | None = None
    window_title: str = "AI Detection Suite"

    def validated(self) -> DetectionConfig:
        if not 0 <= self.camera_index <= 32:
            raise ValueError("camera index must be between 0 and 32")
        if not 160 <= self.capture_width <= 3840:
            raise ValueError("capture width must be between 160 and 3840")
        if not 120 <= self.capture_height <= 2160:
            raise ValueError("capture height must be between 120 and 2160")
        if not 1 <= self.capture_fps <= 240:
            raise ValueError("capture FPS must be between 1 and 240")
        if not 0.25 <= self.processing_scale <= 1.0:
            raise ValueError("processing scale must be between 0.25 and 1.0")
        if not 1 <= self.face_stride <= 10_000 or not 1 <= self.hand_stride <= 10_000:
            raise ValueError("detector strides must be positive")
        if not 0.05 <= self.object_hz <= 60:
            raise ValueError("object update rate must be between 0.05 and 60 Hz")
        if not 0.05 <= self.attribute_hz <= 10:
            raise ValueError("attribute update rate must be between 0.05 and 10 Hz")
        ttl_values = (self.hand_ttl, self.object_ttl, self.attribute_ttl)
        if any(not math.isfinite(value) or not 0 < value <= 60 for value in ttl_values):
            raise ValueError("result TTL values must be finite and in (0, 60]")
        if self.object_img_size < 160 or self.object_img_size > 1280:
            raise ValueError("object image size must be between 160 and 1280")
        if self.object_img_size % 32:
            raise ValueError("object image size must be divisible by 32")
        if not 0 < self.object_confidence <= 1 or not 0 < self.object_iou <= 1:
            raise ValueError("confidence and IoU must be in (0, 1]")
        if not 1 <= self.object_max_detections <= 300:
            raise ValueError("maximum detections must be between 1 and 300")
        if self.hand_backend not in {"auto", "skin", "mediapipe"}:
            raise ValueError("hand backend must be auto, skin, or mediapipe")
        if self.enable_hands and self.hand_backend == "mediapipe" and self.hand_model is None:
            raise ValueError("MediaPipe requires --hand-model FILE.task")
        if self.enable_hands and self.hand_model is not None and self.hand_model_sha256 is None:
            raise ValueError("a hand model requires --hand-model-sha256")
        if self.hand_model_sha256 is not None:
            digest = self.hand_model_sha256.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("hand model SHA-256 must contain 64 hexadecimal characters")
        numeric_device = self.device.isascii() and self.device.isdecimal()
        if self.device not in {"auto", "cpu"} and not numeric_device:
            raise ValueError("device must be auto, cpu, or a numeric GPU index")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max frames must be positive")
        if self.enable_face_attributes and not self.enable_faces:
            raise ValueError("face attributes require face detection")
        if not self.window_title or len(self.window_title) > 128:
            raise ValueError("window title must contain between 1 and 128 characters")
        if any(ord(char) < 32 or ord(char) == 127 for char in self.window_title):
            raise ValueError("window title must not contain control characters")
        return self


def preset_config(name: str) -> DetectionConfig:
    """Return a documented preset without mutating shared global state."""

    base = DetectionConfig()
    presets = {
        "full": base,
        "full-skin": replace(base, hand_backend="skin"),
        "full-mediapipe": replace(base, hand_backend="mediapipe"),
        "full-attributes": replace(base, enable_face_attributes=True),
        "face": replace(base, enable_hands=False, enable_objects=False),
        "face-attributes": replace(
            base,
            enable_hands=False,
            enable_objects=False,
            enable_face_attributes=True,
        ),
        "objects": replace(
            base,
            enable_faces=False,
            enable_hands=False,
            enable_objects=True,
        ),
        "hands": replace(
            base,
            enable_faces=False,
            enable_hands=True,
            enable_objects=False,
        ),
        "hands-skin": replace(
            base,
            enable_faces=False,
            enable_hands=True,
            enable_objects=False,
            hand_backend="skin",
        ),
        "camera": replace(
            base,
            enable_faces=False,
            enable_hands=False,
            enable_objects=False,
        ),
    }
    try:
        return presets[name]
    except KeyError as exc:
        raise ValueError(f"unknown preset: {name}") from exc
