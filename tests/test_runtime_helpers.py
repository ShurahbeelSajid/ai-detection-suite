from __future__ import annotations

import builtins
import contextlib
import io
import unittest
from unittest import mock

from ai_detection_suite import app
from ai_detection_suite.config import DetectionConfig
from ai_detection_suite.domain import Detection, FaceAttributes, HandDetection, InferenceBatch
from ai_detection_suite.rendering import FpsMeter, Renderer
from ai_detection_suite.workers import CapturedError, WorkResult


class _Frame:
    def __init__(self, height: int = 120, width: int = 160) -> None:
        self.shape = (height, width, 3)

    def copy(self) -> _Frame:
        return _Frame(self.shape[0], self.shape[1])

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def __getitem__(self, _key: object) -> _Frame:
        return self


class _Capture:
    def __init__(
        self,
        *,
        opened: bool,
        frame: _Frame | None = None,
        reads: list[tuple[bool, _Frame | None]] | None = None,
    ) -> None:
        self.opened = opened
        self.frame = frame or _Frame()
        self.reads = list(reads or ())
        self.read_calls = 0
        self.released = False
        self.set_calls: list[tuple[int, int]] = []

    def isOpened(self) -> bool:
        return self.opened

    def release(self) -> None:
        self.released = True

    def set(self, setting: int, value: int) -> bool:
        self.set_calls.append((setting, value))
        return True

    def get(self, setting: int) -> float:
        return 160.0 if setting == 1 else 120.0

    def read(self) -> tuple[bool, _Frame | None]:
        self.read_calls += 1
        if self.reads:
            return self.reads.pop(0)
        return True, self.frame


class _CameraCv2:
    CAP_DSHOW = 700
    CAP_MSMF = 1400
    CAP_PROP_FRAME_WIDTH = 1
    CAP_PROP_FRAME_HEIGHT = 2
    CAP_PROP_FPS = 3
    CAP_PROP_BUFFERSIZE = 4

    def __init__(self, captures: list[_Capture]) -> None:
        self.captures = captures
        self.open_calls: list[tuple[object, ...]] = []

    def VideoCapture(self, *arguments: object) -> _Capture:
        self.open_calls.append(arguments)
        return self.captures[len(self.open_calls) - 1]


class _RenderingCv2:
    FONT_HERSHEY_SIMPLEX = 1
    LINE_AA = 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def _record(self, name: str, *arguments: object) -> None:
        self.calls.append((name, arguments))

    def getTextSize(self, text: str, *_arguments: object) -> tuple[tuple[int, int], int]:
        return (max(1, len(text) * 5), 10), 2

    def rectangle(self, *arguments: object) -> None:
        self._record("rectangle", *arguments)

    def putText(self, *arguments: object) -> None:
        self._record("putText", *arguments)

    def line(self, *arguments: object) -> None:
        self._record("line", *arguments)

    def circle(self, *arguments: object) -> None:
        self._record("circle", *arguments)

    def addWeighted(self, *arguments: object) -> None:
        self._record("addWeighted", *arguments)


class AppHelperTests(unittest.TestCase):
    def test_box_and_hand_scaling_use_independent_axes_and_clip(self) -> None:
        self.assertEqual(
            app._scale_box(
                (-5, 20, 120, 80),
                source_width=100,
                source_height=100,
                target_width=200,
                target_height=50,
            ),
            (0, 10, 200, 40),
        )
        scaled = app._scale_hands(
            (
                HandDetection(
                    box=(10, 20, 30, 40),
                    confidence=0.8,
                    landmarks=((0, 0), (100, 100)),
                ),
            ),
            source_width=100,
            source_height=100,
            target_width=200,
            target_height=50,
        )
        self.assertEqual(scaled[0].box, (20, 10, 60, 20))
        self.assertEqual(scaled[0].landmarks, ((0, 0), (199, 49)))

    @staticmethod
    def _work_result(
        batch: InferenceBatch | None,
        *,
        error: CapturedError | None = None,
        sequence: int = 1,
    ) -> WorkResult[InferenceBatch]:
        return WorkResult(
            sequence=sequence,
            frame_id=1,
            captured_at=8.0,
            submitted_at=8.1,
            started_at=8.2,
            finished_at=10.0,
            value=batch,
            error=error,
        )

    def test_fresh_value_honors_success_and_ttl(self) -> None:
        batch = InferenceBatch(
            frame_id=1,
            captured_at=10.0,
            completed_at=10.0,
            items=(),
        )
        result = self._work_result(batch)
        self.assertIs(app._fresh_value(result, now=10.5, ttl=0.5), batch)
        self.assertIsNone(app._fresh_value(result, now=10.5001, ttl=0.5))
        slow_batch = InferenceBatch(
            frame_id=1,
            captured_at=8.0,
            completed_at=10.0,
            items=(),
        )
        self.assertIsNone(app._fresh_value(self._work_result(slow_batch), now=10.1, ttl=1.0))
        self.assertIsNone(app._fresh_value(None, now=10.0, ttl=1.0))
        error = CapturedError("builtins", "ValueError", "bad", "trace")
        self.assertIsNone(app._fresh_value(self._work_result(None, error=error), now=10.0, ttl=1.0))

    def test_face_attributes_are_associated_by_geometry(self) -> None:
        faces = ((0, 0, 10, 10), (30, 0, 40, 10))
        attributes = (
            FaceAttributes(source_box=(31, 0, 41, 10), emotion="second"),
            FaceAttributes(source_box=(1, 0, 11, 10), emotion="first"),
        )
        associated = app._associate_attributes(faces, attributes)
        self.assertEqual(
            [(box, item.emotion) for box, item in associated],
            [((0, 0, 10, 10), "first"), ((30, 0, 40, 10), "second")],
        )

    def test_optional_feature_errors_warn_or_fail_strictly(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            app._optional_feature_error(DetectionConfig(), "objects", ValueError("bad"))
        self.assertIn("objects disabled: bad", output.getvalue())

        with self.assertRaisesRegex(RuntimeError, "objects disabled: bad"):
            app._optional_feature_error(
                DetectionConfig(strict_features=True), "objects", ValueError("bad")
            )

    def test_worker_health_trips_and_success_resets_failure_circuit(self) -> None:
        error = CapturedError("builtins", "RuntimeError", "broken", "trace")
        sequence = -1
        failures = 0
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for current in range(3):
                sequence, failures, disabled = app._worker_health(
                    "object",
                    self._work_result(None, error=error, sequence=current),
                    last_sequence=sequence,
                    failures=failures,
                )
        self.assertTrue(disabled)
        self.assertEqual(failures, 3)
        success = InferenceBatch(4, 4.0, 4.1, ())
        sequence, failures, disabled = app._worker_health(
            "object",
            self._work_result(success, sequence=4),
            last_sequence=sequence,
            failures=failures,
        )
        self.assertEqual((sequence, failures, disabled), (4, 0, False))

    def test_cv2_loader_translates_missing_dependency(self) -> None:
        original_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "cv2":
                raise ImportError("missing")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(RuntimeError, "OpenCV is not installed"):
                app._cv2()


class CameraHelperTests(unittest.TestCase):
    def test_windows_backend_falls_back_and_applies_low_latency_settings(self) -> None:
        directshow = _Capture(opened=False)
        media_foundation = _Capture(opened=True)
        cv2 = _CameraCv2([directshow, media_foundation])
        config = DetectionConfig(
            camera_index=2,
            capture_width=800,
            capture_height=600,
            capture_fps=24,
        )
        with (
            mock.patch.object(app.os, "name", "nt"),
            mock.patch.object(app, "camera_permission_hint", return_value=None),
        ):
            selected = app._open_camera(cv2, config)

        self.assertIs(selected, media_foundation)
        self.assertTrue(directshow.released)
        self.assertEqual(cv2.open_calls, [(2, cv2.CAP_DSHOW), (2, cv2.CAP_MSMF)])
        self.assertEqual(
            media_foundation.set_calls,
            [
                (cv2.CAP_PROP_FRAME_WIDTH, 800),
                (cv2.CAP_PROP_FRAME_HEIGHT, 600),
                (cv2.CAP_PROP_FPS, 24),
                (cv2.CAP_PROP_BUFFERSIZE, 1),
            ],
        )

    def test_windows_camera_falls_back_to_generic_backend(self) -> None:
        directshow = _Capture(opened=False)
        media_foundation = _Capture(opened=False)
        generic = _Capture(opened=True)
        cv2 = _CameraCv2([directshow, media_foundation, generic])
        with (
            mock.patch.object(app.os, "name", "nt"),
            mock.patch.object(app, "camera_permission_hint", return_value=None),
        ):
            selected = app._open_camera(cv2, DetectionConfig(camera_index=1))

        self.assertIs(selected, generic)
        self.assertTrue(directshow.released)
        self.assertTrue(media_foundation.released)
        self.assertEqual(
            cv2.open_calls,
            [(1, cv2.CAP_DSHOW), (1, cv2.CAP_MSMF), (1,)],
        )

    def test_windows_permission_denial_is_rejected_before_video_capture(self) -> None:
        cv2 = _CameraCv2([])
        with (
            mock.patch.object(app.os, "name", "nt"),
            mock.patch.object(
                app,
                "camera_permission_hint",
                return_value="Windows camera access is denied.",
            ),
            self.assertRaisesRegex(RuntimeError, "Windows camera access is denied"),
        ):
            app._open_camera(cv2, DetectionConfig())
        self.assertEqual(cv2.open_calls, [])

    def test_unavailable_camera_is_released_and_reported(self) -> None:
        capture = _Capture(opened=False)
        cv2 = _CameraCv2([capture])
        with mock.patch.object(app.os, "name", "posix"):
            with self.assertRaisesRegex(RuntimeError, "cannot open camera 0"):
                app._open_camera(cv2, DetectionConfig())
        self.assertTrue(capture.released)

    def test_camera_read_retries_transient_failures(self) -> None:
        frame = _Frame()
        capture = _Capture(
            opened=True,
            reads=[(False, None), (True, None), (True, frame)],
        )
        with mock.patch.object(app.time, "sleep") as sleep:
            result = app._read_camera_frame(capture)

        self.assertIs(result, frame)
        self.assertEqual(capture.read_calls, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_camera_read_failure_is_bounded(self) -> None:
        capture = _Capture(
            opened=True,
            reads=[(False, None)] * app.MAX_CAMERA_READ_ATTEMPTS,
        )
        with (
            mock.patch.object(app.time, "sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "consecutive attempts"),
        ):
            app._read_camera_frame(capture)

        self.assertEqual(capture.read_calls, app.MAX_CAMERA_READ_ATTEMPTS)
        self.assertEqual(sleep.call_count, app.MAX_CAMERA_READ_ATTEMPTS - 1)


class RenderingHelperTests(unittest.TestCase):
    def test_fps_meter_reports_current_and_window_average(self) -> None:
        meter = FpsMeter()
        self.assertEqual(meter.tick(10.0), (0.0, 0.0))
        current, average = meter.tick(10.5)
        self.assertAlmostEqual(current, 2.0)
        self.assertAlmostEqual(average, 2.0)
        current, average = meter.tick(10.75)
        self.assertAlmostEqual(current, 4.0)
        self.assertAlmostEqual(average, 2.0 / 0.75)

    def test_renderer_accepts_all_domain_result_types_without_cv2_import(self) -> None:
        cv2 = _RenderingCv2()
        renderer = Renderer(cv2)
        frame = _Frame()
        renderer.faces(frame, ((1, 2, 20, 30),))
        renderer.hands(
            frame,
            (
                HandDetection(
                    box=(5, 5, 30, 40),
                    confidence=0.8,
                    handedness="Right",
                    finger_count=3,
                    landmarks=((5, 5), (6, 6)),
                ),
            ),
        )
        renderer.objects(
            frame,
            (Detection(label="cup", confidence=0.7, box=(10, 10, 40, 50)),),
        )
        renderer.attributes(
            frame,
            (
                (
                    (1, 2, 20, 30),
                    FaceAttributes(
                        source_box=(1, 2, 20, 30),
                        age=30,
                        gender="unspecified",
                        emotion="neutral",
                    ),
                ),
            ),
        )
        renderer.hud(
            frame,
            fps=30.0,
            average_fps=29.0,
            latency_ms=12.0,
            face_count=1,
            hand_count=1,
            object_count=1,
            feature_status="unit test",
            attributes_enabled=True,
        )

        call_names = [name for name, _arguments in cv2.calls]
        self.assertIn("line", call_names)
        self.assertIn("circle", call_names)
        self.assertIn("addWeighted", call_names)
        rendered_text = [arguments[1] for name, arguments in cv2.calls if name == "putText"]
        self.assertIn("Face", rendered_text)
        self.assertIn("cup 0.70", rendered_text)
        self.assertIn("Estimated age: 30", rendered_text)


class CameraOnlyRuntimeTests(unittest.TestCase):
    def test_failed_camera_warmup_releases_capture_before_detector_initialization(self) -> None:
        class FakeCv2:
            def setUseOptimized(self, _enabled: bool) -> None:
                return None

        capture = _Capture(
            opened=True,
            reads=[(False, None)] * app.MAX_CAMERA_READ_ATTEMPTS,
        )
        config = DetectionConfig(
            enable_faces=True,
            enable_hands=False,
            enable_objects=False,
        )
        with (
            mock.patch.object(app, "_cv2", return_value=FakeCv2()),
            mock.patch.object(app, "_open_camera", return_value=capture),
            mock.patch.object(app, "HaarFaceDetector") as face_detector,
            mock.patch.object(app.time, "sleep"),
            self.assertRaisesRegex(RuntimeError, "consecutive attempts"),
        ):
            app.run_app(config)

        face_detector.assert_not_called()
        self.assertTrue(capture.released)

    def test_initialization_failure_closes_already_created_resources(self) -> None:
        class FakeCv2:
            CAP_PROP_FRAME_WIDTH = 1
            CAP_PROP_FRAME_HEIGHT = 2

            def setUseOptimized(self, _enabled: bool) -> None:
                return None

        class FakeHandDetector:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        class FakeWorker:
            def __init__(self) -> None:
                self.stopped = False

            def shutdown(self, **_kwargs: object) -> bool:
                self.stopped = True
                return True

        detector = FakeHandDetector()
        worker = FakeWorker()
        capture = _Capture(opened=True)
        events: list[str] = []

        def open_camera(*_arguments: object) -> _Capture:
            events.append("camera-open")
            return capture

        original_read = capture.read

        def read_frame() -> tuple[bool, _Frame | None]:
            events.append("camera-read")
            return original_read()

        capture.read = read_frame  # type: ignore[method-assign]

        def fail_object_detector(*_arguments: object, **_keywords: object) -> None:
            events.append("object-model")
            raise app.FeatureUnavailable("unavailable")

        config = DetectionConfig(
            enable_faces=False,
            enable_hands=True,
            enable_objects=True,
            hand_backend="skin",
            strict_features=True,
        )
        with (
            mock.patch.object(app, "_cv2", return_value=FakeCv2()),
            mock.patch.object(app, "_open_camera", side_effect=open_camera),
            mock.patch.object(app, "create_hand_detector", return_value=detector),
            mock.patch.object(app, "LatestFrameWorker", return_value=worker),
            mock.patch.object(
                app,
                "YoloObjectDetector",
                side_effect=fail_object_detector,
            ),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "object detection disabled"),
        ):
            app.run_app(config)
        self.assertEqual(events[:3], ["camera-open", "camera-read", "object-model"])
        self.assertTrue(worker.stopped)
        self.assertTrue(detector.closed)
        self.assertTrue(capture.released)

    def test_one_frame_camera_run_cleans_up_without_detectors(self) -> None:
        capture = _Capture(opened=True)

        class FakeCv2:
            CAP_PROP_FRAME_WIDTH = 1
            CAP_PROP_FRAME_HEIGHT = 2
            WINDOW_NORMAL = 3

            def __init__(self) -> None:
                self.destroyed = False
                self.optimized = False

            def setUseOptimized(self, enabled: bool) -> None:
                self.optimized = enabled

            def namedWindow(self, *_arguments: object) -> None:
                return None

            def resizeWindow(self, *_arguments: object) -> None:
                return None

            def imshow(self, *_arguments: object) -> None:
                return None

            def waitKey(self, _delay: int) -> int:
                return -1

            def destroyAllWindows(self) -> None:
                self.destroyed = True

        class RecordingRenderer:
            instance: RecordingRenderer | None = None

            def __init__(self, _cv2: object) -> None:
                self.calls: list[str] = []
                RecordingRenderer.instance = self

            def objects(self, *_arguments: object) -> None:
                self.calls.append("objects")

            def faces(self, *_arguments: object) -> None:
                self.calls.append("faces")

            def hands(self, *_arguments: object) -> None:
                self.calls.append("hands")

            def attributes(self, *_arguments: object) -> None:
                self.calls.append("attributes")

            def hud(self, *_arguments: object, **_keywords: object) -> None:
                self.calls.append("hud")

        cv2 = FakeCv2()
        config = DetectionConfig(
            enable_faces=False,
            enable_hands=False,
            enable_objects=False,
            processing_scale=1.0,
            mirror=False,
            max_frames=1,
        )
        output = io.StringIO()
        with (
            mock.patch.object(app, "_cv2", return_value=cv2),
            mock.patch.object(app, "_open_camera", return_value=capture),
            mock.patch.object(app, "Renderer", RecordingRenderer),
            contextlib.redirect_stdout(output),
        ):
            status = app.run_app(config)

        self.assertEqual(status, 0)
        self.assertTrue(cv2.optimized)
        self.assertTrue(cv2.destroyed)
        self.assertTrue(capture.released)
        self.assertEqual(capture.read_calls, 2)
        self.assertEqual(
            RecordingRenderer.instance.calls,
            ["objects", "faces", "hands", "attributes", "hud"],
        )
        self.assertIn("Active: camera only", output.getvalue())
        self.assertIn("Session complete: 1 frames", output.getvalue())


if __name__ == "__main__":
    unittest.main()
