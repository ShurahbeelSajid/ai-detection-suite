"""Immutable, dependency-free geometry helpers for detection results."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _extent(value: float, name: str, *, allow_zero: bool = True) -> float:
    number = _finite(value, name)
    minimum = 0.0 if allow_zero else math.nextafter(0.0, 1.0)
    if number < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return number


@dataclass(frozen=True, slots=True)
class Box:
    """A half-open ``(x1, y1, x2, y2)`` rectangle."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.x2 < self.x1:
            raise ValueError("x2 must be greater than or equal to x1")
        if self.y2 < self.y1:
            raise ValueError("y2 must be greater than or equal to y1")

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> Box:
        x = _finite(x, "x")
        y = _finite(y, "y")
        width = _extent(width, "width")
        height = _extent(height, "height")
        return cls(x, y, x + width, y + height)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xywh(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.width, self.height

    def clip(self, width: float, height: float) -> Box:
        return clip_box(self, width, height)

    def scale(self, scale_x: float, scale_y: float) -> Box:
        return scale_box(self, scale_x, scale_y)


def clip_box(box: Box, width: float, height: float) -> Box:
    """Clip a box to image bounds, preserving valid zero-area boxes."""

    width = _extent(width, "width")
    height = _extent(height, "height")
    return Box(
        min(max(box.x1, 0.0), width),
        min(max(box.y1, 0.0), height),
        min(max(box.x2, 0.0), width),
        min(max(box.y2, 0.0), height),
    )


def scale_box(box: Box, scale_x: float, scale_y: float) -> Box:
    """Scale X and Y independently instead of assuming a square resize."""

    scale_x = _extent(scale_x, "scale_x")
    scale_y = _extent(scale_y, "scale_y")
    return Box(
        box.x1 * scale_x,
        box.y1 * scale_y,
        box.x2 * scale_x,
        box.y2 * scale_y,
    )


def scale_box_between(
    box: Box,
    *,
    source_width: float,
    source_height: float,
    target_width: float,
    target_height: float,
) -> Box:
    """Scale a box between frames with potentially different aspect ratios."""

    source_width = _extent(source_width, "source_width", allow_zero=False)
    source_height = _extent(source_height, "source_height", allow_zero=False)
    target_width = _extent(target_width, "target_width")
    target_height = _extent(target_height, "target_height")
    return scale_box(
        box,
        target_width / source_width,
        target_height / source_height,
    )


def intersection_over_union(left: Box, right: Box) -> float:
    """Return intersection-over-union, or zero for empty/disjoint boxes."""

    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    union = left.area + right.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


@dataclass(frozen=True, slots=True)
class BoxMatch:
    face_index: int
    attribute_index: int
    iou: float


@dataclass(frozen=True, slots=True)
class FaceAttributeAssociation:
    """One-to-one matches plus both sets of unmatched indices."""

    matches: tuple[BoxMatch, ...]
    unmatched_face_indices: tuple[int, ...]
    unmatched_attribute_indices: tuple[int, ...]


def associate_face_attributes(
    face_boxes: Sequence[Box],
    attribute_boxes: Sequence[Box],
    *,
    min_iou: float = 0.1,
) -> FaceAttributeAssociation:
    """Associate attribute boxes to faces by deterministic greedy IoU.

    Candidates are considered from highest to lowest IoU and each face and
    attribute result may be used once.  The returned matches are ordered by
    face index, independent of detector result ordering.
    """

    min_iou = _finite(min_iou, "min_iou")
    if not 0.0 <= min_iou <= 1.0:
        raise ValueError("min_iou must be between 0 and 1")

    candidates: list[tuple[float, int, int]] = []
    for face_index, face_box in enumerate(face_boxes):
        for attribute_index, attribute_box in enumerate(attribute_boxes):
            score = intersection_over_union(face_box, attribute_box)
            if score > 0.0 and score >= min_iou:
                candidates.append((-score, face_index, attribute_index))
    candidates.sort()

    used_faces: set[int] = set()
    used_attributes: set[int] = set()
    matches: list[BoxMatch] = []
    for negative_score, face_index, attribute_index in candidates:
        if face_index in used_faces or attribute_index in used_attributes:
            continue
        used_faces.add(face_index)
        used_attributes.add(attribute_index)
        matches.append(
            BoxMatch(
                face_index=face_index,
                attribute_index=attribute_index,
                iou=-negative_score,
            )
        )

    matches.sort(key=lambda match: (match.face_index, match.attribute_index))
    return FaceAttributeAssociation(
        matches=tuple(matches),
        unmatched_face_indices=tuple(
            index for index in range(len(face_boxes)) if index not in used_faces
        ),
        unmatched_attribute_indices=tuple(
            index for index in range(len(attribute_boxes)) if index not in used_attributes
        ),
    )
