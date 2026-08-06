"""Small immutable value objects shared by detectors and the renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

Box: TypeAlias = tuple[int, int, int, int]  # x1, y1, x2, y2
Point: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class FramePacket:
    """A captured frame and the timing data needed to reject stale results."""

    frame_id: int
    captured_at: float
    image: Any


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    box: Box


@dataclass(frozen=True, slots=True)
class HandDetection:
    box: Box
    confidence: float | None
    label: str = "Hand candidate"
    handedness: str | None = None
    finger_count: int | None = None
    landmarks: tuple[Point, ...] = ()


@dataclass(frozen=True, slots=True)
class FaceAttributes:
    """Sensitive, probabilistic attributes attached to a source face box."""

    source_box: Box
    age: int | None = None
    gender: str | None = None
    emotion: str | None = None


@dataclass(frozen=True, slots=True)
class FaceAnalysisJob:
    packet: FramePacket
    face_boxes: tuple[Box, ...]


@dataclass(frozen=True, slots=True)
class HandAnalysisJob:
    packet: FramePacket
    face_boxes: tuple[Box, ...]
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class InferenceBatch:
    frame_id: int
    captured_at: float
    completed_at: float
    items: tuple[Any, ...]
    source_width: int | None = None
    source_height: int | None = None
