"""Detector adapters with lazy optional dependencies and stable coordinates."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import default_model_root
from .domain import (
    Box,
    Detection,
    FaceAnalysisJob,
    FaceAttributes,
    FramePacket,
    HandAnalysisJob,
    HandDetection,
    InferenceBatch,
)
from .model_security import ModelSecurityError, validate_local_asset_path, validate_model_path
from .ultralytics_security import UnsafeUltralyticsError, load_secure_yolo_class


class FeatureUnavailable(RuntimeError):
    """Raised when an optional detector cannot be initialized safely."""


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise FeatureUnavailable(
            "OpenCV is not installed. Run: python -m pip install -e ."
        ) from exc
    return cv2


def _clean_label(value: Any, fallback: str) -> str:
    text = str(value).strip()
    text = "".join(char for char in text if char.isprintable() and ord(char) < 128)
    return text[:64] or fallback


def _clip_box(box: Box, width: int, height: int) -> Box | None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


class HaarFaceDetector:
    """Fast CPU face detector using OpenCV's offline Haar cascade."""

    def __init__(self, *, min_size: int = 36) -> None:
        self.cv2 = _load_cv2()
        cascade_path = Path(self.cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = self.cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise FeatureUnavailable(f"failed to load face cascade: {cascade_path}")
        self._min_size = min_size

    def detect(self, frame: np.ndarray) -> tuple[Box, ...]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            return ()
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(self._min_size, self._min_size),
        )
        height, width = frame.shape[:2]
        boxes: list[Box] = []
        for x, y, box_width, box_height in faces:
            clipped = _clip_box(
                (int(x), int(y), int(x + box_width), int(y + box_height)),
                width,
                height,
            )
            if clipped is not None:
                boxes.append(clipped)
        boxes.sort(key=lambda box: (box[0], box[1]))
        return tuple(boxes)


class SkinHandDetector:
    """Low-dependency fallback that reports conservative skin/shape candidates.

    It is intentionally labelled as a candidate detector: color heuristics are not
    equivalent to a trained hand landmark model and vary with lighting and skin tone.
    """

    def __init__(self, *, max_hands: int = 2) -> None:
        self.cv2 = _load_cv2()
        self._kernel = self.cv2.getStructuringElement(self.cv2.MORPH_ELLIPSE, (5, 5))
        self._max_hands = max_hands
        self._hsv_min = np.array((0, 20, 45), dtype=np.uint8)
        self._hsv_max = np.array((25, 255, 255), dtype=np.uint8)
        self._ycrcb_min = np.array((0, 130, 75), dtype=np.uint8)
        self._ycrcb_max = np.array((255, 180, 135), dtype=np.uint8)

    def detect(
        self,
        frame: np.ndarray,
        face_boxes: Sequence[Box] = (),
        *,
        timestamp_ms: int | None = None,
    ) -> tuple[HandDetection, ...]:
        del timestamp_ms
        if frame.ndim != 3 or frame.shape[2] != 3:
            return ()
        frame_height, frame_width = frame.shape[:2]
        frame_area = float(frame_height * frame_width)
        if frame_area <= 0:
            return ()

        hsv = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
        ycrcb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2YCrCb)
        hsv_mask = self.cv2.inRange(hsv, self._hsv_min, self._hsv_max)
        ycrcb_mask = self.cv2.inRange(ycrcb, self._ycrcb_min, self._ycrcb_max)
        mask = self.cv2.bitwise_and(hsv_mask, ycrcb_mask)

        for face_box in face_boxes:
            clipped = _clip_box(face_box, frame_width, frame_height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            margin = max(6, int(0.12 * max(x2 - x1, y2 - y1)))
            self.cv2.rectangle(
                mask,
                (max(0, x1 - margin), max(0, y1 - margin)),
                (min(frame_width, x2 + margin), min(frame_height, y2 + margin)),
                0,
                -1,
            )

        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, self._kernel)
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, self._kernel)
        contours, _ = self.cv2.findContours(
            mask,
            self.cv2.RETR_EXTERNAL,
            self.cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: list[HandDetection] = []
        for contour in contours:
            area = float(self.cv2.contourArea(contour))
            area_ratio = area / frame_area
            if not 0.004 <= area_ratio <= 0.28:
                continue
            x, y, width, height = self.cv2.boundingRect(contour)
            if width < 18 or height < 18:
                continue
            aspect = width / height
            if not 0.25 <= aspect <= 3.5:
                continue
            hull_area = float(self.cv2.contourArea(self.cv2.convexHull(contour)))
            solidity = area / hull_area if hull_area > 0 else 0.0
            extent = area / float(width * height)
            if solidity < 0.45 or extent < 0.25:
                continue
            confidence = min(0.85, 0.25 + area_ratio * 5.0 + solidity * 0.25)
            candidates.append(
                HandDetection(
                    box=(x, y, x + width, y + height),
                    confidence=confidence,
                )
            )

        candidates.sort(
            key=lambda item: (
                item.confidence,
                (item.box[2] - item.box[0]) * (item.box[3] - item.box[1]),
            ),
            reverse=True,
        )
        return tuple(candidates[: self._max_hands])

    def close(self) -> None:
        return None


class MediaPipeTaskHandDetector:
    """MediaPipe Tasks adapter; it never downloads a model at runtime."""

    def __init__(
        self,
        model_path: Path,
        *,
        model_root: Path,
        expected_sha256: str,
        max_hands: int = 2,
        swap_handedness: bool = False,
    ) -> None:
        try:
            path = validate_local_asset_path(
                model_path,
                root=model_root,
                allowed_suffixes=frozenset({".task"}),
                expected_sha256=expected_sha256,
                min_size_bytes=1_024,
                max_size_bytes=100 * 1024 * 1024,
            )
        except (ModelSecurityError, OSError, ValueError) as exc:
            raise FeatureUnavailable(f"untrusted MediaPipe task model: {exc}") from exc

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except (ImportError, AttributeError) as exc:  # pragma: no cover - optional dependency
            raise FeatureUnavailable(
                "MediaPipe Tasks is unavailable; install the hands extra or use --hand-backend skin"
            ) from exc

        try:
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.50,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except (OSError, RuntimeError, ValueError) as exc:
            raise FeatureUnavailable(f"failed to initialize MediaPipe task model: {exc}") from exc
        self._mp = mp
        self._cv2 = _load_cv2()
        self._last_timestamp = -1
        self._swap_handedness = swap_handedness

    @staticmethod
    def _finger_count(landmarks: Sequence[Any], handedness: str | None) -> int:
        if len(landmarks) < 21:
            return 0
        extended = [
            landmarks[8].y < landmarks[6].y,
            landmarks[12].y < landmarks[10].y,
            landmarks[16].y < landmarks[14].y,
            landmarks[20].y < landmarks[18].y,
        ]
        if handedness == "Right":
            thumb = landmarks[4].x < landmarks[3].x
        elif handedness == "Left":
            thumb = landmarks[4].x > landmarks[3].x
        else:
            thumb = abs(landmarks[4].x - landmarks[3].x) > 0.035
        return int(thumb) + sum(extended)

    def detect(
        self,
        frame: np.ndarray,
        face_boxes: Sequence[Box] = (),
        *,
        timestamp_ms: int | None = None,
    ) -> tuple[HandDetection, ...]:
        del face_boxes
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        timestamp = int(timestamp_ms if timestamp_ms is not None else time.monotonic() * 1000)
        timestamp = max(timestamp, self._last_timestamp + 1)
        self._last_timestamp = timestamp
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp)
        frame_height, frame_width = frame.shape[:2]
        hands: list[HandDetection] = []
        for index, landmarks in enumerate(result.hand_landmarks or ()):
            points = tuple(
                (
                    max(0, min(int(point.x * frame_width), frame_width - 1)),
                    max(0, min(int(point.y * frame_height), frame_height - 1)),
                )
                for point in landmarks
            )
            if not points:
                continue
            handedness: str | None = None
            if index < len(result.handedness or ()) and result.handedness[index]:
                category = result.handedness[index][0]
                handedness = _clean_label(category.category_name, "Unknown")
                if self._swap_handedness:
                    handedness = {"Left": "Right", "Right": "Left"}.get(handedness, handedness)
            padding = 10
            box = _clip_box(
                (
                    min(point[0] for point in points) - padding,
                    min(point[1] for point in points) - padding,
                    max(point[0] for point in points) + padding,
                    max(point[1] for point in points) + padding,
                ),
                frame_width,
                frame_height,
            )
            if box is None:
                continue
            hands.append(
                HandDetection(
                    box=box,
                    confidence=None,
                    label="Hand",
                    handedness=handedness,
                    finger_count=self._finger_count(landmarks, handedness),
                    landmarks=points,
                )
            )
        return tuple(hands)

    def close(self) -> None:
        self._landmarker.close()


def create_hand_detector(
    backend: str,
    model_path: Path | None = None,
    *,
    model_root: Path | None = None,
    expected_sha256: str | None = None,
    swap_handedness: bool = False,
) -> Any:
    effective_model_root = default_model_root() if model_root is None else model_root
    if backend == "skin":
        return SkinHandDetector()
    if backend == "mediapipe":
        if model_path is None or expected_sha256 is None:
            raise FeatureUnavailable(
                "MediaPipe Tasks needs --hand-model FILE.task and --hand-model-sha256; "
                "no download is performed"
            )
        return MediaPipeTaskHandDetector(
            model_path,
            model_root=effective_model_root,
            expected_sha256=expected_sha256,
            swap_handedness=swap_handedness,
        )
    if backend == "auto" and model_path is not None:
        if expected_sha256 is None:
            raise FeatureUnavailable("a MediaPipe hand model requires its SHA-256 digest")
        return MediaPipeTaskHandDetector(
            model_path,
            model_root=effective_model_root,
            expected_sha256=expected_sha256,
            swap_handedness=swap_handedness,
        )
    return SkinHandDetector()


class YoloObjectDetector:
    """Single-owner YOLO adapter using verified local checkpoints only."""

    def __init__(
        self,
        model_name: str,
        *,
        image_size: int,
        confidence: float,
        iou: float,
        max_detections: int,
        device: str = "auto",
        allow_unverified: bool = False,
        model_root: Path | None = None,
    ) -> None:
        model_path = validate_model_path(
            model_name,
            root=default_model_root() if model_root is None else model_root,
            allow_unverified=allow_unverified,
        )
        try:
            YOLO = load_secure_yolo_class()
        except UnsafeUltralyticsError as exc:
            raise FeatureUnavailable(str(exc)) from exc

        self._model = YOLO(str(model_path))
        self._image_size = image_size
        self._confidence = confidence
        self._iou = iou
        self._max_detections = max_detections
        self._device = None if device == "auto" else int(device) if device.isdigit() else device

    def __call__(self, packet: FramePacket) -> InferenceBatch:
        kwargs: dict[str, Any] = {
            "source": packet.image,
            "imgsz": self._image_size,
            "conf": self._confidence,
            "iou": self._iou,
            "max_det": self._max_detections,
            "verbose": False,
            "save": False,
            "show": False,
        }
        if self._device is not None:
            kwargs["device"] = self._device
        results = self._model.predict(**kwargs)
        items: list[Detection] = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes):
                box_data = boxes.data.detach().cpu().numpy()
                frame_height, frame_width = packet.image.shape[:2]
                names = self._model.names
                for row in box_data:
                    if len(row) < 6:
                        continue
                    coords = row[:4]
                    score = float(row[4])
                    class_id = int(row[5])
                    clipped = _clip_box(
                        (
                            math.floor(float(coords[0])),
                            math.floor(float(coords[1])),
                            math.ceil(float(coords[2])),
                            math.ceil(float(coords[3])),
                        ),
                        frame_width,
                        frame_height,
                    )
                    if clipped is None:
                        continue
                    if isinstance(names, dict):
                        raw_label = names.get(class_id, f"class-{class_id}")
                    elif 0 <= class_id < len(names):
                        raw_label = names[class_id]
                    else:
                        raw_label = f"class-{class_id}"
                    items.append(
                        Detection(
                            label=_clean_label(raw_label, f"class-{class_id}"),
                            confidence=max(0.0, min(score, 1.0)),
                            box=clipped,
                        )
                    )
        return InferenceBatch(
            frame_id=packet.frame_id,
            captured_at=packet.captured_at,
            completed_at=time.perf_counter(),
            items=tuple(items),
        )


def _deepface_weights_directory() -> Path:
    base = Path(os.environ.get("DEEPFACE_HOME", str(Path.home()))).expanduser()
    return base / ".deepface" / "weights"


class DeepFaceAttributeAnalyzer:
    """Analyze confirmed face crops only; no identity recognition or logging."""

    REQUIRED_WEIGHTS = (
        "age_model_weights.h5",
        "gender_model_weights.h5",
        "facial_expression_model_weights.h5",
    )

    @classmethod
    def _offline_weights_available(cls) -> bool:
        directory = _deepface_weights_directory()
        for name in cls.REQUIRED_WEIGHTS:
            path = directory / name
            try:
                size = path.stat(follow_symlinks=False).st_size
            except (OSError, TypeError):
                return False
            if path.is_symlink() or not path.is_file() or not 1_024 <= size <= 1024**3:
                return False
        return True

    def __init__(self, *, allow_downloads: bool = False, max_faces: int = 2) -> None:
        if not allow_downloads and not self._offline_weights_available():
            raise FeatureUnavailable(
                f"DeepFace attribute weights are not provisioned in "
                f"{_deepface_weights_directory()}. Re-run with --allow-model-downloads "
                "only if you accept its upstream downloads."
            )
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        try:
            from deepface import DeepFace
        except (
            ImportError,
            ValueError,
            RuntimeError,
        ) as exc:  # pragma: no cover - optional dependency
            raise FeatureUnavailable(
                "DeepFace is unavailable. Run: python -m pip install -e .[attributes]"
            ) from exc
        self._deepface = DeepFace
        self._max_faces = max_faces
        self._allow_downloads = allow_downloads

    @staticmethod
    def _expanded_crop(frame: np.ndarray, box: Box) -> np.ndarray | None:
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = box
        margin_x = int((x2 - x1) * 0.12)
        margin_y = int((y2 - y1) * 0.12)
        clipped = _clip_box(
            (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y),
            frame_width,
            frame_height,
        )
        if clipped is None:
            return None
        left, top, right, bottom = clipped
        if right - left < 32 or bottom - top < 32:
            return None
        return np.ascontiguousarray(frame[top:bottom, left:right])

    @staticmethod
    def _normalize_results(raw: Any, expected: int) -> list[dict[str, Any] | None]:
        if isinstance(raw, dict):
            return [raw] + [None] * max(0, expected - 1)
        if not isinstance(raw, list):
            return [None] * expected
        normalized: list[dict[str, Any] | None] = []
        for item in raw:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, list) and item and isinstance(item[0], dict):
                normalized.append(item[0])
            else:
                normalized.append(None)
        return (normalized + [None] * expected)[:expected]

    @staticmethod
    def _safe_age(value: Any) -> int | None:
        try:
            age = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(age) or not 0 <= age <= 120:
            return None
        return round(age)

    def __call__(self, job: FaceAnalysisJob) -> InferenceBatch:
        if not self._allow_downloads and not self._offline_weights_available():
            raise FeatureUnavailable("DeepFace offline weights disappeared before analysis")
        frame = job.packet.image
        ranked_boxes = sorted(
            job.face_boxes,
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
            reverse=True,
        )[: self._max_faces]
        crops: list[np.ndarray] = []
        source_boxes: list[Box] = []
        for box in ranked_boxes:
            crop = self._expanded_crop(frame, box)
            if crop is not None:
                crops.append(crop)
                source_boxes.append(box)
        if not crops:
            return InferenceBatch(
                frame_id=job.packet.frame_id,
                captured_at=job.packet.captured_at,
                completed_at=time.perf_counter(),
                items=(),
            )

        source: Any = crops if len(crops) > 1 else crops[0]
        raw = self._deepface.analyze(
            img_path=source,
            actions=("age", "gender", "emotion"),
            detector_backend="skip",
            enforce_detection=True,
            align=False,
            silent=True,
        )
        parsed = self._normalize_results(raw, len(crops))
        attributes: list[FaceAttributes] = []
        for source_box, result in zip(source_boxes, parsed, strict=True):
            if result is None:
                continue
            gender = result.get("dominant_gender")
            emotion = result.get("dominant_emotion")
            attributes.append(
                FaceAttributes(
                    source_box=source_box,
                    age=self._safe_age(result.get("age")),
                    gender=(_clean_label(gender, "") or None) if gender is not None else None,
                    emotion=(_clean_label(emotion, "") or None) if emotion is not None else None,
                )
            )
        return InferenceBatch(
            frame_id=job.packet.frame_id,
            captured_at=job.packet.captured_at,
            completed_at=time.perf_counter(),
            items=tuple(attributes),
        )


def run_hand_detection(detector: Any, job: HandAnalysisJob) -> InferenceBatch:
    """Worker adapter that keeps hand inference off the capture/display thread."""

    height, width = job.packet.image.shape[:2]
    hands = detector.detect(
        job.packet.image,
        job.face_boxes,
        timestamp_ms=job.timestamp_ms,
    )
    return InferenceBatch(
        frame_id=job.packet.frame_id,
        captured_at=job.packet.captured_at,
        completed_at=time.perf_counter(),
        items=tuple(hands),
        source_width=width,
        source_height=height,
    )
