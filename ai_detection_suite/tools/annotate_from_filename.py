"""Detect object geometry while deriving deterministic classes from filenames."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai_detection_suite.annotations import (
    annotations_to_yolo_text,
    atomic_write_text,
    batched,
    deterministic_class_mapping,
    ensure_targets_writable,
    enumerate_images,
    label_path_for_image,
    parse_filename_label,
)
from ai_detection_suite.config import default_model_root
from ai_detection_suite.model_security import ModelSecurityError, validate_model_path
from ai_detection_suite.ultralytics_security import (
    UnsafeUltralyticsError,
    load_secure_yolo_class,
)

_SAFE_SPLIT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _confidence(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("confidence must be in (0, 1]")
    return parsed


def _split_name(value: str) -> str:
    if not _SAFE_SPLIT.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "split names may contain only letters, numbers, underscores, and hyphens"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    dataset_root = Path.cwd() / "yolo_dataset"
    parser = argparse.ArgumentParser(
        description=(
            "Use YOLO detections as box geometry and the image filename as the "
            "class label. A final numeric suffix is treated as an instance number."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=dataset_root,
        help="dataset containing images/<split> and labels/<split> (default: %(default)s)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=_split_name,
        default=["train", "val"],
        help="dataset splits to annotate (default: train val)",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        help="class manifest path (default: <dataset-root>/labels.txt)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolo11n.pt"),
        help="trusted local YOLO model (default: %(default)s)",
    )
    parser.add_argument("--model-dir", type=Path, default=default_model_root())
    parser.add_argument("--confidence", type=_confidence, default=0.25)
    parser.add_argument("--image-size", type=_positive_int, default=640)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument(
        "--device",
        help="Ultralytics device selection, for example cpu, 0, or 0,1",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace existing labels and the class manifest",
    )
    parser.add_argument(
        "--allow-unverified-model",
        action="store_true",
        help="dangerous: use a custom local checkpoint after path/size checks",
    )
    parser.add_argument("--max-pixels", type=_positive_int, default=50_000_000)
    return parser


def _tensor_values(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _predicted_boxes(result: Any) -> list[Sequence[float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    return _tensor_values(boxes.xyxy)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_root = args.dataset_root.expanduser().resolve()
    classes_file = (
        args.classes_file.expanduser().resolve()
        if args.classes_file
        else dataset_root / "labels.txt"
    )

    jobs: list[tuple[Path, Path, str]] = []
    try:
        for split in args.splits:
            image_directory = dataset_root / "images" / split
            images = enumerate_images(image_directory)
            if not images:
                parser.error(f"no supported images found in split: {image_directory}")
            label_directory = dataset_root / "labels" / split
            jobs.extend(
                (
                    image_path,
                    label_path_for_image(image_path, label_directory),
                    parse_filename_label(image_path),
                )
                for image_path in images
            )
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        parser.error(str(exc))

    class_mapping = deterministic_class_mapping(job[2] for job in jobs)
    targets = [job[1] for job in jobs]
    try:
        ensure_targets_writable(
            [*targets, classes_file],
            overwrite=args.force,
        )
    except (FileExistsError, IsADirectoryError, ValueError) as exc:
        parser.error(str(exc))

    try:
        model_path = validate_model_path(
            args.model,
            root=args.model_dir,
            allow_unverified=args.allow_unverified_model,
        )
    except ModelSecurityError as exc:
        parser.error(str(exc))

    try:
        import cv2
    except ImportError as exc:
        parser.error(f"OpenCV is required to label images: {exc}")
    try:
        YOLO = load_secure_yolo_class()
    except UnsafeUltralyticsError as exc:
        parser.error(f"Ultralytics is required to label images: {exc}")

    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        parser.error(f"could not load model {model_path}: {exc}")

    prediction_options: dict[str, Any] = {
        "conf": args.confidence,
        "imgsz": args.image_size,
        "save": False,
        "stream": False,
        "verbose": False,
    }
    if args.device:
        prediction_options["device"] = args.device

    rendered: list[tuple[Path, str]] = []
    try:
        for job_batch in batched(jobs, args.batch_size):
            frames: list[Any] = []
            dimensions: list[tuple[int, int]] = []
            for image_path, _target, _label in job_batch:
                frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if frame is None or len(frame.shape) < 2:
                    raise RuntimeError(f"could not decode image: {image_path}")
                height, width = frame.shape[:2]
                if width <= 0 or height <= 0:
                    raise RuntimeError(f"image has invalid dimensions: {image_path}")
                if width * height > args.max_pixels:
                    raise RuntimeError(
                        f"image exceeds --max-pixels ({width}x{height}): {image_path}"
                    )
                frames.append(frame)
                dimensions.append((width, height))

            results = list(model.predict(source=frames, batch=len(frames), **prediction_options))
            if len(results) != len(job_batch):
                raise RuntimeError(
                    f"model returned {len(results)} results for {len(job_batch)} images"
                )

            for job, result, (width, height) in zip(job_batch, results, dimensions, strict=True):
                _image_path, target, label = job
                class_id = class_mapping[label]
                annotations = ((class_id, coordinates) for coordinates in _predicted_boxes(result))
                rendered.append((target, annotations_to_yolo_text(annotations, width, height)))
    except Exception as exc:
        parser.error(f"annotation inference failed: {exc}")

    classes_text = "".join(
        f"{class_id}: {label}\n"
        for label, class_id in sorted(class_mapping.items(), key=lambda item: item[1])
    )
    try:
        for target, text in rendered:
            atomic_write_text(target, text, overwrite=args.force)
        atomic_write_text(classes_file, classes_text, overwrite=args.force)
    except (FileExistsError, IsADirectoryError, OSError) as exc:
        parser.error(str(exc))

    summary = ", ".join(
        f"{class_id}:{label}"
        for label, class_id in sorted(class_mapping.items(), key=lambda item: item[1])
    )
    print(f"Annotated {len(rendered)} image(s); classes: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
