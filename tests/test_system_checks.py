from __future__ import annotations

import unittest
from unittest import mock

from ai_detection_suite import system_checks


class SystemCheckTests(unittest.TestCase):
    def test_store_python_package_family_is_extracted(self) -> None:
        prefix = (
            "C:\\Program Files\\WindowsApps\\"
            r"PythonSoftwareFoundation.Python.3.13_3.13.2544.0_x64__qbz5n2kfra8p0"
        )
        self.assertEqual(
            system_checks._store_package_family(prefix),
            "PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0",
        )

    def test_non_windows_permission_state_is_unknown(self) -> None:
        with mock.patch.object(system_checks.os, "name", "posix"):
            self.assertIsNone(system_checks.windows_camera_permission_state())

    def test_denied_camera_permission_produces_actionable_hint(self) -> None:
        with mock.patch.object(
            system_checks,
            "windows_camera_permission_state",
            return_value="deny",
        ):
            hint = system_checks.camera_permission_hint()
        self.assertIsNotNone(hint)
        self.assertIn("Privacy & security > Camera", hint or "")
        self.assertIn("Python", hint or "")


if __name__ == "__main__":
    unittest.main()
