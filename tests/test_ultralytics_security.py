from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

from ai_detection_suite import ultralytics_security


def _fake_modules(*, supported: bool = True, enabled: bool = True) -> dict[str, object]:
    torch = types.ModuleType("torch")
    serialization = types.SimpleNamespace()
    if supported:
        serialization.safe_globals = object()
    torch.serialization = serialization

    tasks = types.ModuleType("ultralytics.nn.tasks")
    tasks._SafeLoad = types.SimpleNamespace(SUPPORTED=supported)
    tasks.SAFE_LOAD = enabled
    nn = types.ModuleType("ultralytics.nn")
    nn.tasks = tasks
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.__path__ = []

    class FakeYOLO:
        pass

    ultralytics.YOLO = FakeYOLO
    return {
        "torch": torch,
        "ultralytics": ultralytics,
        "ultralytics.nn": nn,
        "ultralytics.nn.tasks": tasks,
    }


class SecureUltralyticsLoaderTests(unittest.TestCase):
    def test_rejects_digit_concatenating_prerelease_versions(self) -> None:
        for version in ("8.4.6rc7", "8.4.67rc1"):
            with (
                self.subTest(version=version),
                mock.patch.object(ultralytics_security.metadata, "version", return_value=version),
            ):
                with self.assertRaises(ultralytics_security.UnsafeUltralyticsError):
                    ultralytics_security.load_secure_yolo_class()

    def test_requires_torch_safe_globals_and_active_ultralytics_flag(self) -> None:
        for supported, enabled in ((False, True), (True, False)):
            with (
                self.subTest(supported=supported, enabled=enabled),
                mock.patch.object(ultralytics_security.metadata, "version", return_value="8.4.67"),
                mock.patch.dict(
                    sys.modules,
                    _fake_modules(supported=supported, enabled=enabled),
                ),
                self.assertRaises(ultralytics_security.UnsafeUltralyticsError),
            ):
                ultralytics_security.load_secure_yolo_class()

    def test_returns_yolo_only_when_restricted_loading_is_verified(self) -> None:
        modules = _fake_modules()
        with (
            mock.patch.object(ultralytics_security.metadata, "version", return_value="8.4.67"),
            mock.patch.dict(sys.modules, modules),
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            loaded = ultralytics_security.load_secure_yolo_class()
            self.assertIs(loaded, modules["ultralytics"].YOLO)
            self.assertEqual(os.environ["ULTRALYTICS_SAFE_LOAD"], "true")


if __name__ == "__main__":
    unittest.main()
