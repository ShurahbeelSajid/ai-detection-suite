from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from ai_detection_suite import cli, detectors, diagnostics
from ai_detection_suite.config import (
    DEFAULT_MODEL_ROOT,
    DetectionConfig,
    default_model_root,
    preset_config,
)
from ai_detection_suite.model_security import ModelSecurityError


class DetectionConfigTests(unittest.TestCase):
    def test_default_and_boundary_values_validate(self) -> None:
        with mock.patch.dict(os.environ, {"AI_DETECTION_MODEL_DIR": ""}):
            default = DetectionConfig()
        self.assertIs(default.validated(), default)
        self.assertEqual(default.model_root, DEFAULT_MODEL_ROOT)
        DetectionConfig(
            camera_index=32,
            capture_width=160,
            capture_height=120,
            capture_fps=240,
            processing_scale=1.0,
            object_hz=0.05,
            attribute_hz=10.0,
            object_img_size=1280,
            object_confidence=1.0,
            object_iou=1.0,
            object_max_detections=300,
            device="0",
            max_frames=1,
        ).validated()

    def test_model_root_environment_override_is_dynamic(self) -> None:
        custom_root = Path.home() / "trusted-ai-models"
        with mock.patch.dict(os.environ, {"AI_DETECTION_MODEL_DIR": str(custom_root)}):
            self.assertEqual(default_model_root(), custom_root)
            self.assertEqual(DetectionConfig().model_root, custom_root)

    def test_invalid_values_are_rejected(self) -> None:
        invalid_values = (
            ("camera_index", -1),
            ("camera_index", 33),
            ("capture_width", 159),
            ("capture_height", 119),
            ("capture_fps", 0),
            ("processing_scale", 0.24),
            ("face_stride", 0),
            ("hand_stride", 0),
            ("object_hz", 0.0),
            ("attribute_hz", 10.1),
            ("object_ttl", 0.0),
            ("attribute_ttl", -1.0),
            ("object_img_size", 159),
            ("object_img_size", 385),
            ("object_confidence", 0.0),
            ("object_iou", 1.01),
            ("object_max_detections", 0),
            ("hand_backend", "unknown"),
            ("device", "cuda:0"),
            ("max_frames", 0),
        )
        base = DetectionConfig()
        for field_name, value in invalid_values:
            with self.subTest(field=field_name, value=value):
                with self.assertRaises(ValueError):
                    replace(base, **{field_name: value}).validated()

        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(object_ttl=value), self.assertRaises(ValueError):
                replace(base, object_ttl=value).validated()

    def test_all_documented_presets_have_expected_feature_flags(self) -> None:
        expected = {
            "full": (True, True, True, False, "auto"),
            "full-skin": (True, True, True, False, "skin"),
            "full-mediapipe": (True, True, True, False, "mediapipe"),
            "full-attributes": (True, True, True, True, "auto"),
            "face": (True, False, False, False, "auto"),
            "face-attributes": (True, False, False, True, "auto"),
            "objects": (False, False, True, False, "auto"),
            "hands": (False, True, False, False, "auto"),
            "hands-skin": (False, True, False, False, "skin"),
            "camera": (False, False, False, False, "auto"),
        }
        for name, flags in expected.items():
            with self.subTest(preset=name):
                config = preset_config(name)
                self.assertEqual(
                    (
                        config.enable_faces,
                        config.enable_hands,
                        config.enable_objects,
                        config.enable_face_attributes,
                        config.hand_backend,
                    ),
                    flags,
                )
                if name == "full-mediapipe":
                    with self.assertRaisesRegex(ValueError, "hand-model"):
                        config.validated()
                else:
                    self.assertIs(config.validated(), config)

        with self.assertRaisesRegex(ValueError, "unknown preset"):
            preset_config("missing")


class CliParsingTests(unittest.TestCase):
    def test_parser_applies_overrides_and_boolean_options(self) -> None:
        config, doctor = cli.parse_config(
            [
                "--camera",
                "2",
                "--width",
                "800",
                "--height",
                "600",
                "--fps",
                "24",
                "--processing-scale",
                "0.5",
                "--object-hz",
                "4",
                "--face-attributes",
                "--no-hands",
                "--no-mirror",
                "--hand-model",
                "hand.task",
                "--max-frames",
                "3",
                "--doctor",
            ]
        )
        self.assertTrue(doctor)
        self.assertEqual(config.camera_index, 2)
        self.assertEqual((config.capture_width, config.capture_height), (800, 600))
        self.assertEqual(config.capture_fps, 24)
        self.assertEqual(config.processing_scale, 0.5)
        self.assertEqual(config.object_hz, 4.0)
        self.assertTrue(config.enable_face_attributes)
        self.assertFalse(config.enable_hands)
        self.assertFalse(config.mirror)
        self.assertEqual(config.hand_model, Path("hand.task"))
        self.assertEqual(config.max_frames, 3)

    def test_preset_defaults_and_cross_feature_validation(self) -> None:
        objects, doctor = cli.parse_config([], preset="objects")
        self.assertFalse(doctor)
        self.assertEqual(
            (objects.enable_faces, objects.enable_hands, objects.enable_objects),
            (False, False, True),
        )

        with self.assertRaisesRegex(ValueError, "face attributes require face detection"):
            cli.parse_config(["--face-attributes", "--no-faces"])
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            cli.parse_config(["--img-size", "385"])


class RunCliTests(unittest.TestCase):
    @staticmethod
    def _module(name: str, function_name: str, function: object) -> types.ModuleType:
        module = types.ModuleType(name)
        setattr(module, function_name, function)
        return module

    def test_doctor_path_is_lazy_and_returns_its_status(self) -> None:
        run_doctor = mock.Mock(return_value=7)
        fake_diagnostics = self._module("ai_detection_suite.diagnostics", "run_doctor", run_doctor)
        with mock.patch.dict(sys.modules, {"ai_detection_suite.diagnostics": fake_diagnostics}):
            self.assertEqual(cli.run_cli(["--doctor"], preset="camera"), 7)
        supplied = run_doctor.call_args.kwargs["config"]
        self.assertFalse(supplied.enable_objects)

    def test_app_path_receives_validated_config_and_sensitive_notice(self) -> None:
        run_app = mock.Mock(return_value=9)
        fake_app = self._module("ai_detection_suite.app", "run_app", run_app)
        output = io.StringIO()
        with mock.patch.dict(sys.modules, {"ai_detection_suite.app": fake_app}):
            with contextlib.redirect_stdout(output):
                status = cli.run_cli(["--max-frames", "1"], preset="face-attributes")
        self.assertEqual(status, 9)
        supplied_config = run_app.call_args.args[0]
        self.assertTrue(supplied_config.enable_face_attributes)
        self.assertEqual(supplied_config.max_frames, 1)
        self.assertIn("probabilistic and sensitive", output.getvalue())

    def test_expected_errors_and_interrupts_become_exit_codes(self) -> None:
        for error, expected_status in ((OSError("camera busy"), 2), (KeyboardInterrupt(), 130)):
            with self.subTest(error=type(error).__name__):
                run_app = mock.Mock(side_effect=error)
                fake_app = self._module("ai_detection_suite.app", "run_app", run_app)
                errors = io.StringIO()
                with mock.patch.dict(sys.modules, {"ai_detection_suite.app": fake_app}):
                    with contextlib.redirect_stderr(errors):
                        status = cli.run_cli([], preset="camera")
                self.assertEqual(status, expected_status)
                if expected_status == 2:
                    self.assertIn("error: camera busy", errors.getvalue())

        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assertEqual(cli.run_cli([], preset="missing"), 2)
        self.assertIn("unknown preset", errors.getvalue())


class DiagnosticsTests(unittest.TestCase):
    def _run_doctor(
        self,
        *,
        missing_modules: set[str] = frozenset(),
        model_error: Exception | None = None,
        opencv_distributions: dict[str, str] | None = None,
        opencv_runtime: str = "4.10.0",
        gpu_error: str | None = None,
        config: DetectionConfig | None = None,
        hand_asset_error: Exception | None = None,
        deepface_weights: bool = True,
        camera_permission: str | None = None,
    ) -> tuple[int, str]:
        def find_spec(module_name: str) -> object | None:
            return None if module_name in missing_modules else object()

        def version(distribution: str) -> str:
            versions = {
                "numpy": "2.0",
                "opencv-python": "4.10",
                "ultralytics": "8.4.67",
                "deepface": "0.0.100",
                "mediapipe": "0.10.30",
            }
            try:
                return versions[distribution]
            except KeyError as exc:
                raise diagnostics.metadata.PackageNotFoundError(distribution) from exc

        output = io.StringIO()
        model_effect: object = Path("trusted/yolo11n.pt")
        if model_error is not None:
            model_effect = model_error
        with (
            mock.patch.object(diagnostics.importlib.util, "find_spec", side_effect=find_spec),
            mock.patch.object(diagnostics.metadata, "version", side_effect=version),
            mock.patch.object(diagnostics.sys, "version_info", (3, 12, 0)),
            mock.patch.object(diagnostics.platform, "python_version", return_value="3.12.0"),
            mock.patch.object(
                diagnostics,
                "_opencv_distribution_versions",
                return_value=(
                    {"opencv-python": "4.10"}
                    if opencv_distributions is None
                    else opencv_distributions
                ),
            ),
            mock.patch.object(diagnostics, "_opencv_runtime_version", return_value=opencv_runtime),
            mock.patch.object(diagnostics, "_gpu_device_error", return_value=gpu_error),
            mock.patch.object(
                diagnostics,
                "windows_camera_permission_state",
                return_value=camera_permission,
            ),
            mock.patch.object(
                diagnostics,
                "validate_local_asset_path",
                side_effect=hand_asset_error,
                return_value=Path("trusted/hand.task"),
            ),
            mock.patch.object(
                detectors.DeepFaceAttributeAnalyzer,
                "_offline_weights_available",
                return_value=deepface_weights,
            ),
            mock.patch.object(
                diagnostics,
                "validate_model_path",
                side_effect=model_effect if isinstance(model_effect, Exception) else None,
                return_value=None if isinstance(model_effect, Exception) else model_effect,
            ),
            mock.patch.object(diagnostics, "load_secure_yolo_class", return_value=object()),
            contextlib.redirect_stdout(output),
        ):
            status = diagnostics.run_doctor(config=config)
        return status, output.getvalue()

    def test_doctor_succeeds_with_installed_packages_and_verified_model(self) -> None:
        status, output = self._run_doctor()
        self.assertEqual(status, 0)
        self.assertIn("OpenCV: 4.10 [OK]", output)
        self.assertIn("Ultralytics: 8.4.67 [OK]", output)
        self.assertIn("Model integrity: OK (yolo11n.pt)", output)
        self.assertIn("Camera was not opened", output)

    def test_doctor_fails_for_missing_core_or_bad_model(self) -> None:
        status, output = self._run_doctor(missing_modules={"cv2"})
        self.assertEqual(status, 1)
        self.assertIn("OpenCV: not installed [MISSING]", output)

        status, output = self._run_doctor(model_error=ModelSecurityError("bad digest"))
        self.assertEqual(status, 1)
        self.assertIn("Model integrity: FAILED (bad digest)", output)

    def test_doctor_rejects_conflicting_or_mismatched_opencv(self) -> None:
        status, output = self._run_doctor(
            opencv_distributions={
                "opencv-python": "4.14.0.94",
                "opencv-contrib-python": "5.0.0.93",
            }
        )
        self.assertEqual(status, 1)
        self.assertIn("OpenCV package set: FAILED", output)
        self.assertIn("opencv-contrib-python=5.0.0.93", output)

        status, output = self._run_doctor(opencv_runtime="5.0.0")
        self.assertEqual(status, 1)
        self.assertIn("metadata/runtime mismatch", output)

    def test_doctor_accepts_an_identically_versioned_regular_contrib_pair(self) -> None:
        status, output = self._run_doctor(
            opencv_distributions={
                "opencv-python": "4.14.0.94",
                "opencv-contrib-python": "4.14.0.94",
            },
            opencv_runtime="4.14.0",
        )
        self.assertEqual(status, 0)
        self.assertIn("opencv-contrib-python, opencv-python", output)

    def test_doctor_rejects_an_unavailable_requested_gpu(self) -> None:
        status, output = self._run_doctor(
            config=replace(DetectionConfig(), device="0"),
            gpu_error="CUDA is unavailable in this PyTorch installation",
        )
        self.assertEqual(status, 1)
        self.assertIn("Requested GPU device 0: FAILED", output)

    def test_doctor_checks_configured_optional_model_assets(self) -> None:
        hand_config = replace(
            DetectionConfig(),
            hand_backend="mediapipe",
            hand_model=Path("hand.task"),
            hand_model_sha256="0" * 64,
        )
        status, output = self._run_doctor(
            config=hand_config,
            hand_asset_error=ModelSecurityError("bad hand digest"),
        )
        self.assertEqual(status, 1)
        self.assertIn("MediaPipe task integrity: FAILED (bad hand digest)", output)

        attribute_config = replace(DetectionConfig(), enable_face_attributes=True)
        status, output = self._run_doctor(
            config=attribute_config,
            deepface_weights=False,
        )
        self.assertEqual(status, 1)
        self.assertIn("DeepFace offline weights: FAILED", output)

    def test_camera_permission_denial_is_reported_without_opening_the_camera(self) -> None:
        status, output = self._run_doctor(camera_permission="deny")
        self.assertEqual(status, 0)
        self.assertIn("Camera permission: BLOCKED", output)


if __name__ == "__main__":
    unittest.main()
