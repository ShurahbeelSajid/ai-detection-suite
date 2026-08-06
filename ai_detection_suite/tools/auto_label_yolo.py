"""Auto-label images with the class IDs predicted by a local YOLO model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai_detection_suite.annotations import (
    annotations_to_yolo_text,
    atomic_write_text,
    batched,
    ensure_targets_writable,
    enumerate_images,
    label_path_for_image,
)
from ai_detection_suite.config import default_model_root
from ai_detection_suite.model_security import ModelSecurityError, validate_model_path
from ai_detection_suite.ultralytics_security import (
    UnsafeUltralyticsError,
    load_secure_yolo_class,
)


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


def build_parser() -> argparse.ArgumentParser:
    dataset_root = Path.cwd() / "yolo_dataset"
    parser = argparse.ArgumentParser(
        description=(
            "Create YOLO labels from a trusted local model. Existing labels are "
            "never replaced unless --force is supplied."
        )
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=dataset_root / "images" / "train",
        help="input image directory (default: %(default)s)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=dataset_root / "labels" / "train",
        help="output label directory (default: %(default)s)",
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
        help="atomically replace existing label files",
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


def _predicted_annotations(result: Any) -> list[tuple[int, Sequence[float]]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    coordinates = _tensor_values(boxes.xyxy)
    class_ids = _tensor_values(boxes.cls)
    if len(coordinates) != len(class_ids):
        raise RuntimeError("model returned mismatched boxes and class IDs")
    return [
        (int(class_id), coordinate)
        for class_id, coordinate in zip(class_ids, coordinates, strict=True)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        images = enumerate_images(args.images)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        parser.error(str(exc))
    if not images:
        parser.error(f"no supported images found in: {args.images}")

    targets = [label_path_for_image(image, args.labels) for image in images]
    try:
        ensure_targets_writable(targets, overwrite=args.force)
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

    rendered: list[tuple[Path, str]] = []
    prediction_options: dict[str, Any] = {
        "conf": args.confidence,
        "imgsz": args.image_size,
        "save": False,
        "stream": False,
        "verbose": False,
    }
    if args.device:
        prediction_options["device"] = args.device

    try:
        for image_batch in batched(images, args.batch_size):
            frames: list[Any] = []
            dimensions: list[tuple[int, int]] = []
            for image_path in image_batch:
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
            if len(results) != len(image_batch):
                raise RuntimeError(
                    f"model returned {len(results)} results for {len(image_batch)} images"
                )

            for image_path, result, (width, height) in zip(
                image_batch, results, dimensions, strict=True
            ):
                text = annotations_to_yolo_text(_predicted_annotations(result), width, height)
                rendered.append((label_path_for_image(image_path, args.labels), text))
    except Exception as exc:
        parser.error(f"annotation inference failed: {exc}")

    try:
        for target, text in rendered:
            atomic_write_text(target, text, overwrite=args.force)
    except (FileExistsError, IsADirectoryError, OSError) as exc:
        parser.error(str(exc))

    print(f"Annotated {len(rendered)} image(s) in {Path(args.labels).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
