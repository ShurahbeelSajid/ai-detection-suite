"""OpenCV rendering kept separate from inference and synchronization."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .domain import Box, Detection, FaceAttributes, HandDetection

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass(slots=True)
class FpsMeter:
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    _last_at: float | None = None

    def tick(self, now: float) -> tuple[float, float]:
        current = 0.0
        if self._last_at is not None:
            elapsed = now - self._last_at
            if elapsed > 0:
                current = 1.0 / elapsed
                self.samples.append(elapsed)
        self._last_at = now
        average = len(self.samples) / sum(self.samples) if self.samples else current
        return current, average


class Renderer:
    def __init__(self, cv2_module: Any) -> None:
        self.cv2 = cv2_module
        self.font = cv2_module.FONT_HERSHEY_SIMPLEX

    def _blend_panel(
        self,
        frame: Any,
        left: int,
        top: int,
        right: int,
        bottom: int,
        *,
        alpha: float,
    ) -> None:
        """Blend only the affected ROI instead of copying the entire frame."""

        if right <= left or bottom <= top:
            return
        roi = frame[top:bottom, left:right]
        if getattr(roi, "size", 1) == 0:
            return
        overlay = roi.copy()
        self.cv2.rectangle(
            overlay,
            (0, 0),
            (right - left - 1, bottom - top - 1),
            (0, 0, 0),
            -1,
        )
        self.cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

    def _label(
        self,
        frame: Any,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        *,
        scale: float = 0.5,
    ) -> None:
        height, width = frame.shape[:2]
        text = text[:96]
        (text_width, text_height), baseline = self.cv2.getTextSize(text, self.font, scale, 1)
        x = max(0, min(origin[0], max(0, width - text_width - 6)))
        y = max(text_height + 5, min(origin[1], height - baseline - 2))
        self.cv2.rectangle(
            frame,
            (x, y - text_height - 5),
            (x + text_width + 6, y + baseline + 2),
            (0, 0, 0),
            -1,
        )
        self.cv2.putText(frame, text, (x + 3, y), self.font, scale, color, 1, self.cv2.LINE_AA)

    def faces(self, frame: Any, boxes: Sequence[Box]) -> None:
        for x1, y1, x2, y2 in boxes:
            self.cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
            self._label(frame, "Face", (x1, y1 - 5), (255, 180, 40))

    def hands(self, frame: Any, hands: Sequence[HandDetection]) -> None:
        for hand in hands:
            x1, y1, x2, y2 = hand.box
            for start, end in HAND_CONNECTIONS:
                if start < len(hand.landmarks) and end < len(hand.landmarks):
                    self.cv2.line(
                        frame,
                        hand.landmarks[start],
                        hand.landmarks[end],
                        (70, 210, 70),
                        1,
                        self.cv2.LINE_AA,
                    )
            for point in hand.landmarks:
                self.cv2.circle(frame, point, 2, (30, 255, 30), -1)
            self.cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)
            parts = [hand.handedness or hand.label]
            if hand.finger_count is not None:
                parts.append(f"fingers {hand.finger_count}")
            if hand.confidence is not None:
                parts.append(f"{hand.confidence:.2f}")
            self._label(frame, " | ".join(parts), (x1, y1 - 5), (80, 255, 80))

    def objects(self, frame: Any, objects: Sequence[Detection]) -> None:
        for item in objects:
            x1, y1, x2, y2 = item.box
            color = (0, 230, 255) if item.confidence >= 0.65 else (0, 170, 255)
            self.cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            self._label(
                frame,
                f"{item.label} {item.confidence:.2f}",
                (x1, y1 - 5),
                color,
            )

    def attributes(
        self,
        frame: Any,
        associated: Iterable[tuple[Box, FaceAttributes]],
    ) -> None:
        frame_height, frame_width = frame.shape[:2]
        for current_box, item in associated:
            x1, y1, _x2, y2 = current_box
            lines: list[str] = []
            if item.age is not None:
                lines.append(f"Estimated age: {item.age}")
            if item.gender:
                lines.append(f"Predicted gender: {item.gender}")
            if item.emotion:
                lines.append(f"Expression: {item.emotion}")
            if not lines:
                continue
            panel_width = min(250, frame_width)
            panel_height = 20 + 18 * len(lines)
            panel_x = max(0, min(x1, frame_width - panel_width))
            panel_y = y1 - panel_height - 8
            if panel_y < 0:
                panel_y = min(frame_height - panel_height, y2 + 8)
            if panel_y < 0:
                continue
            self._blend_panel(
                frame,
                panel_x,
                panel_y,
                panel_x + panel_width,
                panel_y + panel_height,
                alpha=0.78,
            )
            self.cv2.rectangle(
                frame,
                (panel_x, panel_y),
                (panel_x + panel_width, panel_y + panel_height),
                (220, 100, 220),
                1,
            )
            for index, text in enumerate(lines):
                self.cv2.putText(
                    frame,
                    text,
                    (panel_x + 6, panel_y + 18 + index * 18),
                    self.font,
                    0.44,
                    (255, 230, 255),
                    1,
                    self.cv2.LINE_AA,
                )

    def hud(
        self,
        frame: Any,
        *,
        fps: float,
        average_fps: float,
        latency_ms: float,
        face_count: int,
        hand_count: int,
        object_count: int,
        feature_status: str,
        attributes_enabled: bool,
    ) -> None:
        width = max(1, min(490, frame.shape[1] - 8))
        height = 94 if attributes_enabled else 78
        self._blend_panel(frame, 4, 4, width, height, alpha=0.75)
        self.cv2.putText(
            frame,
            f"FPS {fps:.1f} (avg {average_fps:.1f}) | processing {latency_ms:.1f} ms",
            (10, 24),
            self.font,
            0.48,
            (100, 255, 100),
            1,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            frame,
            f"Faces {face_count} | Hands {hand_count} | Objects {object_count}",
            (10, 44),
            self.font,
            0.48,
            (255, 255, 255),
            1,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            frame,
            feature_status[:90],
            (10, 64),
            self.font,
            0.42,
            (100, 220, 255),
            1,
            self.cv2.LINE_AA,
        )
        if attributes_enabled:
            self.cv2.putText(
                frame,
                "Face attributes are probabilistic; never use for high-stakes decisions.",
                (10, 84),
                self.font,
                0.38,
                (180, 180, 255),
                1,
                self.cv2.LINE_AA,
            )
