from __future__ import annotations

import re
import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from ai_detection_suite import __version__, cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ImportSurfaceTests(unittest.TestCase):
    def test_all_non_main_modules_import_without_optional_ai_dependencies(self) -> None:
        package_modules = (
            "ai_detection_suite",
            "ai_detection_suite.annotations",
            "ai_detection_suite.app",
            "ai_detection_suite.cli",
            "ai_detection_suite.config",
            "ai_detection_suite.detectors",
            "ai_detection_suite.diagnostics",
            "ai_detection_suite.domain",
            "ai_detection_suite.geometry",
            "ai_detection_suite.model_security",
            "ai_detection_suite.rendering",
            "ai_detection_suite.system_checks",
            "ai_detection_suite.tools",
            "ai_detection_suite.tools.annotate_from_filename",
            "ai_detection_suite.tools.auto_label_yolo",
            "ai_detection_suite.ultralytics_security",
            "ai_detection_suite.workers",
        )
        modules = package_modules
        blocked = {
            "cv2",
            "deepface",
            "mediapipe",
            "tensorflow",
            "tf_keras",
            "torch",
            "ultralytics",
        }
        script = f"""
import builtins
import importlib

modules = {modules!r}
blocked = {blocked!r}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.partition('.')[0] in blocked:
        raise AssertionError('eager optional dependency import: ' + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
for module_name in modules:
    importlib.import_module(module_name)
print(len(modules))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(int(completed.stdout.strip()), len(modules))

    def test_package_version_is_semantic(self) -> None:
        self.assertIsNotNone(re.fullmatch(r"\d+\.\d+\.\d+", __version__))


class EntrypointTests(unittest.TestCase):
    def test_package_main_delegates_to_cli(self) -> None:
        run_cli = mock.Mock(return_value=23)
        sys.modules.pop("ai_detection_suite.__main__", None)
        with mock.patch.object(cli, "run_cli", run_cli):
            with self.assertRaises(SystemExit) as raised:
                runpy.run_module(
                    "ai_detection_suite",
                    run_name="__main__",
                    alter_sys=True,
                )
        self.assertEqual(raised.exception.code, 23)
        run_cli.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
