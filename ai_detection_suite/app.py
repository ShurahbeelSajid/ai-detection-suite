"""Canonical low-latency webcam runtime."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from typing import Any

from .config import DetectionConfig
from .detectors import (
    DeepFaceAttributeAnalyzer,
    FeatureUnavailable,
    HaarFaceDetector,
    SkinHandDetector,
    YoloObjectDetector,
    create_hand_detector,
    run_hand_detection,
)
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
from .geometry import Box as GeometryBox
from .geometry import associate_face_attributes, scale_box_between
from .rendering import FpsMeter, Renderer
from .system_checks import camera_permission_hint
from .workers import LatestFrameWorker, WorkerClosedError, WorkResult

MAX_CONSECUTIVE_WORKER_FAILURES = 3
MAX_CAMERA_READ_ATTEMPTS = 5
CAMERA_READ_RETRY_DELAY_SECONDS = 0.02


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed. Run: python -m pip install -e .") from exc
    return cv2


def _scale_box(
    box: Box,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> Box:
    scaled = scale_box_between(
        GeometryBox(*box),
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
    ).clip(target_width, target_height)
    return (
        round(scaled.x1),
        round(scaled.y1),
        round(scaled.x2),
        round(scaled.y2),
    )


def _scale_hands(
    hands: Sequence[HandDetection],
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[HandDetection, ...]:
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return tuple(
        replace(
            hand,
            box=_scale_box(
                hand.box,
                source_width=source_width,
                source_height=source_height,
                target_width=target_width,
                target_height=target_height,
            ),
            landmarks=tuple(
                (
                    max(0, min(round(x * scale_x), target_width - 1)),
                    max(0, min(round(y * scale_y), target_height - 1)),
                )
                for x, y in hand.landmarks
            ),
        )
        for hand in hands
    )


def _fresh_value(
    result: WorkResult[InferenceBatch] | None,
    *,
    now: float,
    ttl: float,
) -> InferenceBatch | None:
    if result is None or not result.succeeded or result.value is None:
        return None
    if now - result.value.captured_at > ttl:
        return None
    return result.value


def _associate_attributes(
    face_boxes: Sequence[Box],
    attributes: Sequence[FaceAttributes],
) -> tuple[tuple[Box, FaceAttributes], ...]:
    if not face_boxes or not attributes:
        return ()
    association = associate_face_attributes(
        tuple(GeometryBox(*box) for box in face_boxes),
        tuple(GeometryBox(*item.source_box) for item in attributes),
        min_iou=0.12,
    )
    return tuple(
        (face_boxes[match.face_index], attributes[match.attribute_index])
        for match in association.matches
    )


def _open_camera(cv2: Any, config: DetectionConfig) -> Any:
    if os.name == "nt":
        permission_hint = camera_permission_hint()
        if permission_hint is not None:
            raise RuntimeError(f"cannot open camera {config.camera_index}; {permission_hint}")

    capture = None
    if os.name == "nt":
        for backend_name in ("CAP_DSHOW", "CAP_MSMF"):
            backend = getattr(cv2, backend_name, None)
            if backend is None:
                continue
            capture = cv2.VideoCapture(config.camera_index, backend)
            if capture.isOpened():
                break
            capture.release()
            capture = None
    if capture is None:
        capture = cv2.VideoCapture(config.camera_index)
    if not capture.isOpened():
        capture.release()
        hint = camera_permission_hint()
        detail = hint or "check permissions and close other apps that may be using it."
        raise RuntimeError(f"cannot open camera {config.camera_index}; {detail}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_height)
    capture.set(cv2.CAP_PROP_FPS, config.capture_fps)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _read_camera_frame(
    capture: Any,
    *,
    max_attempts: int = MAX_CAMERA_READ_ATTEMPTS,
) -> Any:
    """Return one valid frame after a bounded number of transient failures."""

    if max_attempts < 1:
        raise ValueError("camera read attempts must be positive")
    for attempt in range(1, max_attempts + 1):
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame
        if attempt < max_attempts:
            time.sleep(CAMERA_READ_RETRY_DELAY_SECONDS)
    raise RuntimeError(
        f"camera did not return a frame after {max_attempts} consecutive attempts"
    )


def _optional_feature_error(config: DetectionConfig, feature: str, exc: Exception) -> None:
    message = f"{feature} disabled: {exc}"
    if config.strict_features:
        raise RuntimeError(message) from exc
    print(f"WARNING: {message}")


def _worker_health(
    feature: str,
    result: WorkResult[InferenceBatch] | None,
    *,
    last_sequence: int,
    failures: int,
) -> tuple[int, int, bool]:
    """Report each completed worker result once and trip after repeated failures."""

    if result is None or result.sequence == last_sequence:
        return last_sequence, failures, False
    if result.error is None:
        return result.sequence, 0, False
    failures += 1
    print(
        f"WARNING: {feature} inference failed ({failures}/"
        f"{MAX_CONSECUTIVE_WORKER_FAILURES}): "
        f"{result.error.type_name}: {result.error.message}"
    )
    return (
        result.sequence,
        failures,
        failures >= MAX_CONSECUTIVE_WORKER_FAILURES,
    )


def _feature_status(features: Sequence[str]) -> str:
    return " | ".join(features) if features else "camera only"


def run_app(config: DetectionConfig) -> int:
    config = config.validated()
    cv2 = _cv2()
    cv2.setUseOptimized(True)

    face_detector: HaarFaceDetector | None = None
    hand_detector: Any = None
    hand_worker: LatestFrameWorker[HandAnalysisJob, InferenceBatch] | None = None
    hand_worker_resource: LatestFrameWorker[HandAnalysisJob, InferenceBatch] | None = None
    object_worker: LatestFrameWorker[FramePacket, InferenceBatch] | None = None
    attribute_worker: LatestFrameWorker[FaceAnalysisJob, InferenceBatch] | None = None
    all_workers: list[LatestFrameWorker[Any, InferenceBatch]] = []
    capture = None
    window_created = False
    frame_id = 0

    try:
        capture = _open_camera(cv2, config)
        # Confirm the device can deliver data before loading expensive models.
        # This warm-up frame may be several seconds old after model startup, so
        # deliberately discard it and fetch a fresh frame in the main loop.
        _read_camera_frame(capture)
        reported_width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        reported_height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        negotiated_width = reported_width if reported_width > 0 else config.capture_width
        negotiated_height = reported_height if reported_height > 0 else config.capture_height
        print(f"Camera: {negotiated_width}x{negotiated_height}")

        active_features: list[str] = []
        hand_feature_name: str | None = None

        if config.enable_faces:
            face_detector = HaarFaceDetector()
            active_features.append("Haar faces")

        if config.enable_hands:
            try:
                hand_detector = create_hand_detector(
                    config.hand_backend,
                    config.hand_model,
                    model_root=config.model_root,
                    expected_sha256=config.hand_model_sha256,
                    swap_handedness=not config.mirror,
                )
            except FeatureUnavailable as exc:
                _optional_feature_error(config, "MediaPipe hands", exc)
                if config.hand_backend == "auto":
                    try:
                        hand_detector = SkinHandDetector()
                    except FeatureUnavailable as fallback_exc:
                        _optional_feature_error(config, "skin hand fallback", fallback_exc)
            if hand_detector is not None:
                hand_feature_name = (
                    "MediaPipe hands"
                    if hand_detector.__class__.__name__.startswith("MediaPipe")
                    else "skin candidates"
                )
                hand_worker = LatestFrameWorker(
                    partial(run_hand_detection, hand_detector),
                    name="hand-worker",
                    daemon=True,
                )
                hand_worker_resource = hand_worker
                all_workers.append(hand_worker)
                active_features.append(hand_feature_name)

        if config.enable_objects:
            try:
                object_detector = YoloObjectDetector(
                    config.object_model,
                    image_size=config.object_img_size,
                    confidence=config.object_confidence,
                    iou=config.object_iou,
                    max_detections=config.object_max_detections,
                    device=config.device,
                    allow_unverified=config.allow_unverified_model,
                    model_root=config.model_root,
                )
                object_worker = LatestFrameWorker(
                    object_detector,
                    name="yolo-object-worker",
                    daemon=True,
                )
                all_workers.append(object_worker)
                active_features.append("YOLO objects")
            except (FeatureUnavailable, OSError, ValueError, RuntimeError) as exc:
                _optional_feature_error(config, "object detection", exc)

        if config.enable_face_attributes:
            try:
                attribute_analyzer = DeepFaceAttributeAnalyzer(
                    allow_downloads=config.allow_model_downloads
                )
                attribute_worker = LatestFrameWorker(
                    attribute_analyzer,
                    name="deepface-attribute-worker",
                    daemon=True,
                )
                all_workers.append(attribute_worker)
                active_features.append("DeepFace attributes")
            except (FeatureUnavailable, OSError, ValueError, RuntimeError) as exc:
                _optional_feature_error(config, "face attributes", exc)

        feature_status = _feature_status(active_features)
        print(f"Active: {feature_status}")
        print("No frames or detections are saved. Press Q or Esc to quit; R resets FPS.")

        renderer = Renderer(cv2)
        fps_meter = FpsMeter()
        cached_processing_faces: tuple[Box, ...] = ()
        cached_hands: tuple[HandDetection, ...] = ()
        displayed_hand_sequence = -1
        hand_result_sequence = -1
        object_result_sequence = -1
        attribute_result_sequence = -1
        hand_failures = 0
        object_failures = 0
        attribute_failures = 0
        next_object_at = 0.0
        next_attribute_at = 0.0

        cv2.namedWindow(config.window_title, cv2.WINDOW_NORMAL)
        window_created = True
        cv2.resizeWindow(config.window_title, negotiated_width, negotiated_height)

        while True:
            frame = _read_camera_frame(capture)
            captured_at = time.perf_counter()
            frame_id += 1
            if config.mirror:
                frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            full_height, full_width = frame.shape[:2]

            if config.processing_scale == 1.0:
                processing_frame = frame
            else:
                processing_width = max(1, round(full_width * config.processing_scale))
                processing_height = max(1, round(full_height * config.processing_scale))
                processing_frame = cv2.resize(
                    frame,
                    (processing_width, processing_height),
                    interpolation=cv2.INTER_AREA,
                )
            processing_height, processing_width = processing_frame.shape[:2]

            if face_detector is not None and (frame_id == 1 or frame_id % config.face_stride == 0):
                cached_processing_faces = face_detector.detect(processing_frame)
            face_boxes = tuple(
                _scale_box(
                    box,
                    source_width=processing_width,
                    source_height=processing_height,
                    target_width=full_width,
                    target_height=full_height,
                )
                for box in cached_processing_faces
            )

            now = time.perf_counter()
            full_packet = FramePacket(frame_id=frame_id, captured_at=captured_at, image=frame)
            if hand_worker is not None and (frame_id == 1 or frame_id % config.hand_stride == 0):
                processing_packet = FramePacket(
                    frame_id=frame_id,
                    captured_at=captured_at,
                    image=processing_frame,
                )
                hand_job = HandAnalysisJob(
                    packet=processing_packet,
                    face_boxes=cached_processing_faces,
                    timestamp_ms=round(captured_at * 1000),
                )
                try:
                    hand_worker.submit(hand_job, frame_id=frame_id, captured_at=captured_at)
                except WorkerClosedError:
                    hand_worker = None
                    cached_hands = ()
            if object_worker is not None and now >= next_object_at:
                next_object_at = now + 1.0 / config.object_hz
                try:
                    object_worker.submit(
                        full_packet,
                        frame_id=frame_id,
                        captured_at=captured_at,
                    )
                except WorkerClosedError:
                    object_worker = None
            if attribute_worker is not None and face_boxes and now >= next_attribute_at:
                next_attribute_at = now + 1.0 / config.attribute_hz
                try:
                    attribute_worker.submit(
                        FaceAnalysisJob(packet=full_packet, face_boxes=face_boxes),
                        frame_id=frame_id,
                        captured_at=captured_at,
                    )
                except WorkerClosedError:
                    attribute_worker = None

            hand_result = hand_worker.latest_result if hand_worker is not None else None
            object_result = object_worker.latest_result if object_worker is not None else None
            attribute_result = (
                attribute_worker.latest_result if attribute_worker is not None else None
            )

            hand_result_sequence, hand_failures, disable_hands = _worker_health(
                "hand",
                hand_result,
                last_sequence=hand_result_sequence,
                failures=hand_failures,
            )
            object_result_sequence, object_failures, disable_objects = _worker_health(
                "object",
                object_result,
                last_sequence=object_result_sequence,
                failures=object_failures,
            )
            attribute_result_sequence, attribute_failures, disable_attributes = _worker_health(
                "attribute",
                attribute_result,
                last_sequence=attribute_result_sequence,
                failures=attribute_failures,
            )

            for should_disable, feature_name in (
                (disable_hands, hand_feature_name),
                (disable_objects, "YOLO objects"),
                (disable_attributes, "DeepFace attributes"),
            ):
                if should_disable and config.strict_features:
                    raise RuntimeError(f"{feature_name or 'optional'} inference failed repeatedly")

            if disable_hands and hand_worker is not None:
                hand_worker.shutdown(wait=False)
                hand_worker = None
                cached_hands = ()
                if hand_feature_name in active_features:
                    active_features.remove(hand_feature_name)
            if disable_objects and object_worker is not None:
                object_worker.shutdown(wait=False)
                object_worker = None
                if "YOLO objects" in active_features:
                    active_features.remove("YOLO objects")
            if disable_attributes and attribute_worker is not None:
                attribute_worker.shutdown(wait=False)
                attribute_worker = None
                if "DeepFace attributes" in active_features:
                    active_features.remove("DeepFace attributes")
            if disable_hands or disable_objects or disable_attributes:
                feature_status = _feature_status(active_features)
                print(f"Active: {feature_status}")

            now = time.perf_counter()
            hand_batch = _fresh_value(hand_result, now=now, ttl=config.hand_ttl)
            object_batch = _fresh_value(object_result, now=now, ttl=config.object_ttl)
            attribute_batch = _fresh_value(
                attribute_result,
                now=now,
                ttl=config.attribute_ttl,
            )

            if hand_batch is None:
                cached_hands = ()
            elif hand_result is not None and hand_result.sequence != displayed_hand_sequence:
                source_width = hand_batch.source_width
                source_height = hand_batch.source_height
                if source_width and source_height:
                    processing_hands = tuple(
                        item for item in hand_batch.items if isinstance(item, HandDetection)
                    )
                    cached_hands = _scale_hands(
                        processing_hands,
                        source_width=source_width,
                        source_height=source_height,
                        target_width=full_width,
                        target_height=full_height,
                    )
                    displayed_hand_sequence = hand_result.sequence

            objects = (
                tuple(item for item in object_batch.items if isinstance(item, Detection))
                if object_batch is not None
                else ()
            )
            attributes = (
                tuple(item for item in attribute_batch.items if isinstance(item, FaceAttributes))
                if attribute_batch is not None
                else ()
            )

            renderer.objects(display_frame, objects)
            renderer.faces(display_frame, face_boxes)
            renderer.hands(display_frame, cached_hands)
            renderer.attributes(display_frame, _associate_attributes(face_boxes, attributes))
            current_fps, average_fps = fps_meter.tick(now)
            renderer.hud(
                display_frame,
                fps=current_fps,
                average_fps=average_fps,
                latency_ms=(time.perf_counter() - captured_at) * 1000.0,
                face_count=len(face_boxes),
                hand_count=len(cached_hands),
                object_count=len(objects),
                feature_status=feature_status,
                attributes_enabled=attribute_worker is not None,
            )
            cv2.imshow(config.window_title, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                fps_meter = FpsMeter()
            if config.max_frames is not None and frame_id >= config.max_frames:
                break
    finally:
        worker_stopped: dict[int, bool] = {}
        for worker in all_workers:
            try:
                stopped = worker.shutdown(wait=True, timeout=3.0)
            except Exception as exc:
                stopped = False
                print(f"WARNING: worker cleanup failed: {exc}")
            worker_stopped[id(worker)] = stopped
            if not stopped:
                print(
                    "WARNING: inference worker did not stop within 3 seconds; "
                    "its daemon thread will finish in the background"
                )
        hand_worker_stopped = hand_worker_resource is None or worker_stopped.get(
            id(hand_worker_resource), False
        )
        if hand_detector is not None and hand_worker_stopped:
            try:
                hand_detector.close()
            except Exception as exc:
                print(f"WARNING: hand detector cleanup failed: {exc}")
        elif hand_detector is not None:
            print("WARNING: deferred hand detector cleanup because inference is still running")
        if capture is not None:
            try:
                capture.release()
            except Exception as exc:
                print(f"WARNING: camera cleanup failed: {exc}")
        if window_created:
            try:
                cv2.destroyAllWindows()
            except Exception as exc:
                print(f"WARNING: window cleanup failed: {exc}")

    print(f"Session complete: {frame_id} frames")
    return 0
