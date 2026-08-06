from __future__ import annotations

import builtins
import contextlib
import importlib
import io
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from ai_detection_suite.annotations import (
    annotations_to_yolo_text,
    atomic_write_text,
    batched,
    deterministic_class_mapping,
    ensure_targets_writable,
    enumerate_images,
    normalize_class_label,
    parse_filename_label,
    xyxy_to_normalized,
)


class _FakeFrame:
    shape = (50, 100, 3)


class _FakeTensor:
    def __init__(self, values: object) -> None:
        self._values = values

    def tolist(self) -> object:
        return self._values


class _FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = _FakeTensor([[0, 0, 50, 25], [50, 25, 120, 60]])
        self.cls = _FakeTensor([2, 7])

    def __len__(self) -> int:
        return 2


class _FakeResult:
    boxes = _FakeBoxes()


class _FakeYOLO:
    prediction_calls: list[int] = []

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def predict(self, *, source: list[object], **_options: object) -> list[_FakeResult]:
        self.prediction_calls.append(len(source))
        return [_FakeResult() for _frame in source]


def _fake_ai_modules(read_paths: list[Path]) -> dict[str, types.ModuleType]:
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1  # type: ignore[attr-defined]

    def imread(path: str, _mode: int) -> _FakeFrame:
        read_paths.append(Path(path))
        return _FakeFrame()

    cv2.imread = imread  # type: ignore[attr-defined]
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    return {"cv2": cv2, "ultralytics": ultralytics}


class ClassMappingTests(unittest.TestCase):
    def test_filename_parser_preserves_multiword_classes(self) -> None:
        self.assertEqual(parse_filename_label("cell_phone_001.JPG"), "cell_phone")
        self.assertEqual(parse_filename_label("Red Cup-12.png"), "red_cup")
        self.assertEqual(parse_filename_label("archive.v2.jpeg"), "archive.v2")

    def test_filename_parser_rejects_empty_class(self) -> None:
        with self.assertRaises(ValueError):
            parse_filename_label("___001.jpg")

    def test_class_mapping_is_normalized_and_order_independent(self) -> None:
        expected = {"cat": 0, "dog": 1, "red_fox": 2}
        self.assertEqual(
            deterministic_class_mapping(["Red Fox", "Dog", "cat", "DOG"]),
            expected,
        )
        self.assertEqual(
            deterministic_class_mapping(reversed(["Red Fox", "Dog", "cat", "DOG"])),
            expected,
        )

    def test_class_labels_reject_path_and_control_characters(self) -> None:
        with self.assertRaises(ValueError):
            normalize_class_label("animals/cat")
        with self.assertRaises(ValueError):
            normalize_class_label("cat\x00dog")


class ImageEnumerationTests(unittest.TestCase):
    def test_enumeration_is_filtered_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "z.PNG").write_bytes(b"not decoded in this unit test")
            (root / "A.jpg").write_bytes(b"not decoded in this unit test")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "inside.jpg").write_bytes(b"ignore nested files")

            self.assertEqual(
                [path.name for path in enumerate_images(root)],
                ["A.jpg", "z.PNG"],
            )

    def test_missing_and_non_directory_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(FileNotFoundError):
                enumerate_images(root / "missing")
            regular_file = root / "image.jpg"
            regular_file.write_bytes(b"x")
            with self.assertRaises(NotADirectoryError):
                enumerate_images(regular_file)


class BoxConversionTests(unittest.TestCase):
    def test_boxes_are_clipped_then_normalized(self) -> None:
        converted = xyxy_to_normalized((-10, 5, 120, 60), 100, 50)
        self.assertIsNotNone(converted)
        assert converted is not None
        for actual, expected in zip(converted, (0.5, 0.55, 1.0, 0.9), strict=False):
            self.assertAlmostEqual(actual, expected)

    def test_degenerate_and_outside_boxes_are_omitted(self) -> None:
        self.assertIsNone(xyxy_to_normalized((10, 10, 10, 20), 100, 50))
        self.assertIsNone(xyxy_to_normalized((150, 10, 200, 20), 100, 50))
        self.assertIsNone(xyxy_to_normalized((20, 20, 10, 30), 100, 50))

    def test_nonfinite_coordinates_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            xyxy_to_normalized((0, 0, math.inf, 10), 100, 50)

    def test_multiple_boxes_are_serialized_without_overwriting_each_other(self) -> None:
        text = annotations_to_yolo_text(
            [
                (3, (0, 0, 100, 50)),
                (4, (50, 10, 150, 40)),
                (5, (100, 0, 100, 10)),
            ],
            100,
            50,
        )
        self.assertEqual(
            text,
            "3 0.500000 0.500000 1.000000 1.000000\n4 0.750000 0.500000 0.500000 0.600000\n",
        )


class OutputSafetyTests(unittest.TestCase):
    def test_atomic_writer_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "labels" / "sample.txt"
            atomic_write_text(target, "first\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "first\n")

            with self.assertRaises(FileExistsError):
                atomic_write_text(target, "second\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "first\n")

            atomic_write_text(target, "second\n", overwrite=True)
            self.assertEqual(target.read_text(encoding="utf-8"), "second\n")
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_preflight_rejects_existing_and_duplicate_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "sample.txt"
            target.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_targets_writable([target], overwrite=False)
            self.assertEqual(ensure_targets_writable([target], overwrite=True), [target])

            duplicate = root / "." / "new.txt"
            with self.assertRaises(ValueError):
                ensure_targets_writable([root / "new.txt", duplicate], overwrite=True)

    def test_batching_is_bounded_and_validated(self) -> None:
        self.assertEqual(
            [list(batch) for batch in batched([1, 2, 3, 4, 5], 2)],
            [[1, 2], [3, 4], [5]],
        )
        with self.assertRaises(ValueError):
            list(batched([1], 0))


class CliIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeYOLO.prediction_calls.clear()

    def test_tool_model_defaults_honor_environment_override(self) -> None:
        from ai_detection_suite.tools import annotate_from_filename, auto_label_yolo

        custom_root = Path.home() / "trusted-tool-models"
        with mock.patch.dict(os.environ, {"AI_DETECTION_MODEL_DIR": str(custom_root)}):
            self.assertEqual(auto_label_yolo.build_parser().parse_args([]).model_dir, custom_root)
            self.assertEqual(
                annotate_from_filename.build_parser().parse_args([]).model_dir,
                custom_root,
            )

    def test_tool_dataset_defaults_follow_parser_construction_cwd(self) -> None:
        from ai_detection_suite.tools import annotate_from_filename, auto_label_yolo

        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            changed_cwd = Path(temporary_directory).resolve()
            try:
                os.chdir(changed_cwd)
                auto_args = auto_label_yolo.build_parser().parse_args([])
                filename_args = annotate_from_filename.build_parser().parse_args([])
            finally:
                os.chdir(original_cwd)

        dataset_root = changed_cwd / "yolo_dataset"
        self.assertEqual(auto_args.images, dataset_root / "images" / "train")
        self.assertEqual(auto_args.labels, dataset_root / "labels" / "train")
        self.assertEqual(filename_args.dataset_root, dataset_root)

    def test_auto_label_batches_and_reads_each_image_once(self) -> None:
        from ai_detection_suite.tools import auto_label_yolo

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            for name in ("first.jpg", "second.png", "third.jpeg"):
                (images / name).write_bytes(b"fake image")
            model = root / "model.pt"
            model.write_bytes(b"fake model")
            read_paths: list[Path] = []

            with (
                mock.patch.dict(sys.modules, _fake_ai_modules(read_paths)),
                mock.patch.object(auto_label_yolo, "validate_model_path", return_value=model),
                mock.patch.object(
                    auto_label_yolo, "load_secure_yolo_class", return_value=_FakeYOLO
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = auto_label_yolo.main(
                    [
                        "--images",
                        str(images),
                        "--labels",
                        str(labels),
                        "--model",
                        str(model),
                        "--batch-size",
                        "2",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertEqual(len(read_paths), 3)
            self.assertEqual(len(set(read_paths)), 3)
            self.assertEqual(_FakeYOLO.prediction_calls, [2, 1])
            expected = (
                "2 0.250000 0.250000 0.500000 0.500000\n7 0.750000 0.750000 0.500000 0.500000\n"
            )
            for name in ("first.txt", "second.txt", "third.txt"):
                self.assertEqual((labels / name).read_text(encoding="utf-8"), expected)

            error_output = io.StringIO()
            with contextlib.redirect_stderr(error_output):
                with self.assertRaises(SystemExit) as raised:
                    auto_label_yolo.main(
                        [
                            "--images",
                            str(images),
                            "--labels",
                            str(labels),
                            "--model",
                            str(model),
                        ]
                    )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("--force", error_output.getvalue())
            self.assertEqual(len(read_paths), 3)

    def test_filename_cli_uses_one_mapping_across_the_requested_split(self) -> None:
        from ai_detection_suite.tools import annotate_from_filename

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images" / "train"
            images.mkdir(parents=True)
            (images / "Dog_2.jpg").write_bytes(b"fake image")
            (images / "cat_1.png").write_bytes(b"fake image")
            model = root / "model.pt"
            model.write_bytes(b"fake model")
            read_paths: list[Path] = []

            with (
                mock.patch.dict(sys.modules, _fake_ai_modules(read_paths)),
                mock.patch.object(
                    annotate_from_filename, "validate_model_path", return_value=model
                ),
                mock.patch.object(
                    annotate_from_filename, "load_secure_yolo_class", return_value=_FakeYOLO
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = annotate_from_filename.main(
                    [
                        "--dataset-root",
                        str(root),
                        "--splits",
                        "train",
                        "--model",
                        str(model),
                        "--batch-size",
                        "2",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertEqual(len(read_paths), 2)
            self.assertEqual(len(set(read_paths)), 2)
            self.assertEqual((root / "labels.txt").read_text(encoding="utf-8"), "0: cat\n1: dog\n")
            cat_text = (root / "labels" / "train" / "cat_1.txt").read_text(encoding="utf-8")
            dog_text = (root / "labels" / "train" / "Dog_2.txt").read_text(encoding="utf-8")
            self.assertTrue(cat_text.startswith("0 "))
            self.assertTrue(dog_text.startswith("1 "))
            self.assertEqual(len(cat_text.splitlines()), 2)
            self.assertEqual(len(dog_text.splitlines()), 2)


class ImportSafetyTests(unittest.TestCase):
    def test_cli_modules_do_not_import_ai_dependencies(self) -> None:
        original_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name.partition(".")[0] in {"cv2", "ultralytics"}:
                raise AssertionError(f"eager AI dependency import: {name}")
            return original_import(name, *args, **kwargs)

        for module_name in (
            "ai_detection_suite.tools.auto_label_yolo",
            "ai_detection_suite.tools.annotate_from_filename",
        ):
            sys.modules.pop(module_name, None)
            with mock.patch("builtins.__import__", side_effect=guarded_import):
                module = importlib.import_module(module_name)
            self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
