"""Dependency-free tests for the local model trust boundary."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_detection_suite import model_security


class ModelSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()
        self.payload = b"known model bytes"
        self.model = self.root / "known.pt"
        self.model.write_bytes(self.payload)
        self.allowlist = {
            "known.pt": model_security.ModelAllowlistEntry(
                size_bytes=len(self.payload),
                sha256=hashlib.sha256(self.payload).hexdigest(),
            )
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_valid_allowlisted_relative_path_is_anchored_to_injected_root(self) -> None:
        outside_cwd = tempfile.TemporaryDirectory()
        original_cwd = Path.cwd()
        try:
            os.chdir(outside_cwd.name)
            validated = model_security.validate_model_path(
                "known.pt",
                root=self.root,
                allowlist=self.allowlist,
            )
        finally:
            os.chdir(original_cwd)
            outside_cwd.cleanup()
        self.assertEqual(validated, self.model.resolve())

    def test_tilde_expands_in_model_and_root_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary_directory:
            root = Path(temporary_directory)
            payload = b"home model"
            model = root / "known.pt"
            model.write_bytes(payload)
            allowlist = {
                "known.pt": model_security.ModelAllowlistEntry(
                    size_bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            }
            home_relative_root = Path("~") / root.name
            validated = model_security.validate_model_path(
                home_relative_root / model.name,
                root=home_relative_root,
                allowlist=allowlist,
            )
        self.assertEqual(validated, model.resolve())

    def test_binary_asset_requires_trusted_path_size_suffix_and_digest(self) -> None:
        payload = b"task-model" * 256
        asset = self.root / "hand.task"
        asset.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        validated = model_security.validate_local_asset_path(
            "hand.task",
            root=self.root,
            allowed_suffixes=frozenset({".task"}),
            expected_sha256=digest,
            min_size_bytes=1024,
            max_size_bytes=4096,
        )
        self.assertEqual(validated, asset)
        with self.assertRaises(model_security.ModelIntegrityError):
            model_security.validate_local_asset_path(
                asset,
                root=self.root,
                allowed_suffixes=frozenset({".task"}),
                expected_sha256="0" * 64,
                min_size_bytes=1024,
                max_size_bytes=4096,
            )

    def test_digest_comparison_uses_constant_time_primitive(self) -> None:
        with mock.patch.object(
            model_security.hmac,
            "compare_digest",
            wraps=model_security.hmac.compare_digest,
        ) as compare_digest:
            model_security.validate_model_path(
                self.model,
                root=self.root,
                allowlist=self.allowlist,
            )
        compare_digest.assert_called_once_with(
            hashlib.sha256(self.payload).hexdigest(),
            self.allowlist["known.pt"].sha256,
        )

    def test_unlisted_model_fails_closed(self) -> None:
        custom_model = self.root / "custom.pt"
        custom_model.write_bytes(b"custom")
        with self.assertRaises(model_security.UnverifiedModelError):
            model_security.validate_model_path(
                custom_model,
                root=self.root,
                allowlist=self.allowlist,
            )

    def test_explicit_escape_hatch_allows_unlisted_regular_pt_file(self) -> None:
        custom_model = self.root / "custom.pt"
        custom_model.write_bytes(b"custom")
        validated = model_security.validate_model_path(
            custom_model,
            root=self.root,
            allowlist=self.allowlist,
            allow_unverified=True,
        )
        self.assertEqual(validated, custom_model.resolve())

    def test_escape_hatch_does_not_bypass_known_hash_mismatch(self) -> None:
        self.model.write_bytes(b"tampered model bytes")
        tampered_allowlist = {
            "known.pt": model_security.ModelAllowlistEntry(
                size_bytes=len(b"tampered model bytes"),
                sha256=hashlib.sha256(b"different same length!").hexdigest(),
            )
        }
        with self.assertRaises(model_security.ModelIntegrityError):
            model_security.validate_model_path(
                self.model,
                root=self.root,
                allowlist=tampered_allowlist,
                allow_unverified=True,
            )

    def test_known_size_mismatch_is_rejected_before_hashing(self) -> None:
        wrong_size_allowlist = {
            "known.pt": model_security.ModelAllowlistEntry(
                size_bytes=len(self.payload) + 1,
                sha256=hashlib.sha256(self.payload).hexdigest(),
            )
        }
        with mock.patch.object(model_security, "sha256_file") as sha256_file:
            with self.assertRaises(model_security.ModelIntegrityError):
                model_security.validate_model_path(
                    self.model,
                    root=self.root,
                    allowlist=wrong_size_allowlist,
                )
        sha256_file.assert_not_called()

    def test_unsupported_suffix_is_rejected_even_when_unverified_is_allowed(self) -> None:
        unsafe_file = self.root / "custom.pkl"
        unsafe_file.write_bytes(b"pickle")
        with self.assertRaises(model_security.ModelPathError):
            model_security.validate_model_path(
                unsafe_file,
                root=self.root,
                allowlist={},
                allow_unverified=True,
            )

    def test_empty_and_oversized_files_are_rejected(self) -> None:
        empty_model = self.root / "empty.pt"
        empty_model.touch()
        with self.assertRaises(model_security.ModelPathError):
            model_security.validate_model_path(
                empty_model,
                root=self.root,
                allowlist={},
                allow_unverified=True,
            )

        oversized_model = self.root / "oversized.pt"
        oversized_model.write_bytes(b"1234")
        with self.assertRaises(model_security.ModelPathError):
            model_security.validate_model_path(
                oversized_model,
                root=self.root,
                allowlist={},
                allow_unverified=True,
                max_size_bytes=3,
            )

    def test_absolute_and_parent_traversal_outside_root_are_rejected(self) -> None:
        outside_directory = tempfile.TemporaryDirectory()
        try:
            outside_model = Path(outside_directory.name) / "outside.pt"
            outside_model.write_bytes(b"outside")
            for supplied_path in (outside_model, self.root / ".." / outside_model.name):
                with self.subTest(supplied_path=supplied_path):
                    with self.assertRaises(model_security.ModelPathError):
                        model_security.validate_model_path(
                            supplied_path,
                            root=self.root,
                            allowlist={},
                            allow_unverified=True,
                        )
        finally:
            outside_directory.cleanup()

    def test_symbolic_link_model_is_rejected(self) -> None:
        link = self.root / "linked.pt"
        try:
            link.symlink_to(self.model)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"symbolic links are unavailable: {exc}")
        with self.assertRaises(model_security.ModelPathError):
            model_security.validate_model_path(
                link,
                root=self.root,
                allowlist={},
                allow_unverified=True,
            )

    def test_default_allowlist_matches_manifest_identity_fields(self) -> None:
        manifest_path = model_security.DEFAULT_MODEL_ROOT / "model-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_models = {item["filename"]: item for item in manifest["models"]}

        self.assertEqual(set(model_security.DEFAULT_MODEL_ALLOWLIST), set(manifest_models))
        for filename, expected in model_security.DEFAULT_MODEL_ALLOWLIST.items():
            with self.subTest(filename=filename):
                self.assertEqual(expected.size_bytes, manifest_models[filename]["size_bytes"])
                self.assertEqual(expected.sha256, manifest_models[filename]["sha256"])

    def test_bundled_model_exists_and_validates_from_default_root(self) -> None:
        with mock.patch.dict(os.environ, {"AI_DETECTION_MODEL_DIR": ""}):
            validated = model_security.validate_model_path("yolo11n.pt")
        self.assertEqual(validated, (model_security.DEFAULT_MODEL_ROOT / "yolo11n.pt").resolve())


if __name__ == "__main__":
    unittest.main()
