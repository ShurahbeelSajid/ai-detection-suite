"""Dependency-free helpers for producing safe, deterministic YOLO labels."""

from __future__ import annotations

import math
import operator
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TypeAlias, TypeVar

DEFAULT_IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})

Coordinates: TypeAlias = tuple[float, float, float, float]
Annotation: TypeAlias = tuple[int, Sequence[float]]

_INSTANCE_SUFFIX = re.compile(r"[._-]\d+$")
_WHITESPACE = re.compile(r"\s+")
_REPEATED_UNDERSCORES = re.compile(r"_+")
_T = TypeVar("_T")


def normalize_class_label(label: str) -> str:
    """Return one stable class spelling suitable for a class manifest."""

    if not isinstance(label, str):
        raise TypeError("class label must be a string")

    normalized = unicodedata.normalize("NFKC", label).strip().casefold()
    normalized = _WHITESPACE.sub("_", normalized)
    normalized = _REPEATED_UNDERSCORES.sub("_", normalized)
    normalized = normalized.strip("._-")

    if not normalized:
        raise ValueError("class label cannot be empty")
    if "/" in normalized or "\\" in normalized:
        raise ValueError("class label cannot contain path separators")
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise ValueError("class label cannot contain control characters")
    return normalized


def parse_filename_label(path: str | os.PathLike[str]) -> str:
    """Extract a class from names such as ``cell_phone_012.jpg``.

    Only a final numeric instance suffix is removed. This preserves meaningful
    underscores in multi-word class names and avoids the old first-token rule.
    """

    stem = Path(path).stem
    without_instance = _INSTANCE_SUFFIX.sub("", stem)
    return normalize_class_label(without_instance)


def deterministic_class_mapping(labels: Iterable[str]) -> dict[str, int]:
    """Map normalized, unique labels to IDs independent of input order."""

    canonical_labels = {normalize_class_label(label) for label in labels}
    if not canonical_labels:
        raise ValueError("at least one class label is required")
    return {label: index for index, label in enumerate(sorted(canonical_labels))}


def enumerate_images(
    directory: str | os.PathLike[str],
    extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> list[Path]:
    """Return supported regular images in deterministic order.

    Symbolic links and entries resolving outside ``directory`` are ignored so
    annotation jobs cannot be redirected to unexpected files.
    """

    image_directory = Path(directory).expanduser()
    if not image_directory.exists():
        raise FileNotFoundError(f"image directory does not exist: {image_directory}")
    if not image_directory.is_dir():
        raise NotADirectoryError(f"image path is not a directory: {image_directory}")

    normalized_extensions = {
        extension.casefold() if extension.startswith(".") else f".{extension.casefold()}"
        for extension in extensions
    }
    if not normalized_extensions:
        raise ValueError("at least one image extension is required")

    root = image_directory.resolve(strict=True)
    images: list[Path] = []
    for candidate in image_directory.iterdir():
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if candidate.suffix.casefold() not in normalized_extensions:
            continue

        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        images.append(resolved)

    return sorted(images, key=lambda path: (path.name.casefold(), path.name))


def xyxy_to_normalized(
    box: Sequence[float],
    image_width: int,
    image_height: int,
) -> Coordinates | None:
    """Clip an XYXY box to the image and convert it to normalized YOLO form.

    Boxes with no remaining area after clipping are omitted by returning
    ``None``. Non-finite coordinates and invalid image sizes are rejected.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if len(box) != 4:
        raise ValueError("an XYXY box must contain exactly four coordinates")

    left, top, right, bottom = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        raise ValueError("box coordinates must be finite")

    left = min(max(left, 0.0), float(image_width))
    right = min(max(right, 0.0), float(image_width))
    top = min(max(top, 0.0), float(image_height))
    bottom = min(max(bottom, 0.0), float(image_height))
    if right <= left or bottom <= top:
        return None

    width = right - left
    height = bottom - top
    return (
        ((left + right) / 2.0) / image_width,
        ((top + bottom) / 2.0) / image_height,
        width / image_width,
        height / image_height,
    )


def annotations_to_yolo_text(
    annotations: Iterable[Annotation],
    image_width: int,
    image_height: int,
    *,
    precision: int = 6,
) -> str:
    """Serialize class/XYXY pairs, clipping invalid-area boxes away."""

    if not 0 <= precision <= 12:
        raise ValueError("precision must be between 0 and 12")

    lines: list[str] = []
    for class_id_value, box in annotations:
        try:
            class_id = operator.index(class_id_value)
        except TypeError as exc:
            raise TypeError("class ID must be an integer") from exc
        if class_id < 0:
            raise ValueError("class ID cannot be negative")

        normalized = xyxy_to_normalized(box, image_width, image_height)
        if normalized is None:
            continue
        values = " ".join(f"{value:.{precision}f}" for value in normalized)
        lines.append(f"{class_id} {values}\n")
    return "".join(lines)


def label_path_for_image(
    image_path: str | os.PathLike[str],
    label_directory: str | os.PathLike[str],
) -> Path:
    """Return the conventional YOLO label target for an image."""

    return Path(label_directory) / f"{Path(image_path).stem}.txt"


def ensure_targets_writable(
    targets: Iterable[str | os.PathLike[str]],
    *,
    overwrite: bool,
) -> list[Path]:
    """Reject duplicate or existing output targets before expensive inference."""

    paths = [Path(target) for target in targets]
    canonical: dict[str, Path] = {}
    for path in paths:
        key = os.path.normcase(os.path.abspath(os.fspath(path)))
        if key in canonical:
            raise ValueError(f"multiple inputs resolve to the same label: {path}")
        canonical[key] = path
        if path.is_dir():
            raise IsADirectoryError(f"label target is a directory: {path}")

    if not overwrite:
        existing = [path for path in paths if os.path.lexists(path)]
        if existing:
            preview = ", ".join(str(path) for path in existing[:3])
            if len(existing) > 3:
                preview += f", and {len(existing) - 3} more"
            raise FileExistsError(
                f"refusing to overwrite existing annotation output(s): {preview}; "
                "pass --force to replace them"
            )
    return paths


def atomic_write_text(
    target: str | os.PathLike[str],
    text: str,
    *,
    overwrite: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """Atomically publish text without replacing an existing file by default."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir():
        raise IsADirectoryError(f"text target is a directory: {destination}")
    if not overwrite and os.path.lexists(destination):
        raise FileExistsError(
            f"refusing to overwrite existing file: {destination}; pass --force to replace it"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        if overwrite:
            os.replace(temporary, destination)
        else:
            # A hard link publishes the completed inode atomically and fails if
            # another process created the destination after our preflight.
            os.link(temporary, destination)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def batched(items: Sequence[_T], batch_size: int) -> Iterator[Sequence[_T]]:
    """Yield bounded slices without copying or importing an AI framework."""

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
