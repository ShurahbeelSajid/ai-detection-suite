"""Fail-closed loader for Ultralytics' restricted checkpoint mode."""

from __future__ import annotations

import os
from importlib import metadata
from typing import Any

MIN_SAFE_ULTRALYTICS = "8.4.67"


class UnsafeUltralyticsError(RuntimeError):
    """Raised when restricted checkpoint loading cannot be guaranteed."""


def load_secure_yolo_class() -> Any:
    """Return ``ultralytics.YOLO`` only when restricted loading is active.

    Ultralytics snapshots its environment flag during import, so setting the
    flag after another library imported Ultralytics is not sufficient. This
    function checks the imported runtime state as well as the installed
    versions and fails closed if either guarantee is unavailable.
    """

    os.environ["ULTRALYTICS_SAFE_LOAD"] = "true"

    try:
        from packaging.version import InvalidVersion, Version
    except ImportError as exc:  # pragma: no cover - declared core dependency
        raise UnsafeUltralyticsError("packaging is required for safe version checks") from exc

    try:
        installed = metadata.version("ultralytics")
    except metadata.PackageNotFoundError as exc:
        raise UnsafeUltralyticsError(
            "Ultralytics is not installed. Run: python -m pip install -e .[objects]"
        ) from exc

    try:
        installed_version = Version(installed)
    except InvalidVersion as exc:
        raise UnsafeUltralyticsError(
            f"cannot verify the installed Ultralytics version: {installed!r}"
        ) from exc
    minimum_version = Version(MIN_SAFE_ULTRALYTICS)
    if installed_version.is_prerelease or installed_version < minimum_version:
        raise UnsafeUltralyticsError(
            f"Ultralytics {installed} is not an accepted restricted-loader release; "
            f"install >= {MIN_SAFE_ULTRALYTICS}"
        )

    try:
        import torch
    except ImportError as exc:
        raise UnsafeUltralyticsError(
            "PyTorch >=2.5 is required for restricted checkpoint loading"
        ) from exc
    if not hasattr(torch.serialization, "safe_globals"):
        raise UnsafeUltralyticsError(
            "this PyTorch build lacks torch.serialization.safe_globals; install PyTorch >=2.5"
        )

    try:
        from ultralytics import YOLO
        from ultralytics.nn import tasks as ultralytics_tasks
    except (ImportError, RuntimeError) as exc:
        raise UnsafeUltralyticsError(f"failed to import Ultralytics safely: {exc}") from exc

    safe_loader = getattr(ultralytics_tasks, "_SafeLoad", None)
    supported = bool(safe_loader and getattr(safe_loader, "SUPPORTED", False))
    enabled = getattr(ultralytics_tasks, "SAFE_LOAD", False) is True
    if not supported or not enabled:
        raise UnsafeUltralyticsError(
            "Ultralytics was imported before restricted loading was enabled; "
            "restart this Python process and load the suite before Ultralytics"
        )
    return YOLO


__all__ = [
    "MIN_SAFE_ULTRALYTICS",
    "UnsafeUltralyticsError",
    "load_secure_yolo_class",
]
