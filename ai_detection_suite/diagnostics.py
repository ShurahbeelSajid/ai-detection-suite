"""Side-effect-free installation and model integrity diagnostics."""

from __future__ import annotations

import importlib.util
import platform
import sys
from importlib import metadata

from .config import DetectionConfig
from .model_security import (
    ModelSecurityError,
    validate_local_asset_path,
    validate_model_path,
)
from .system_checks import windows_camera_permission_state
from .ultralytics_security import UnsafeUltralyticsError, load_secure_yolo_class

PACKAGES = {
    "numpy": ("numpy", "numpy"),
    "OpenCV": ("cv2", "opencv-python"),
    "Ultralytics": ("ultralytics", "ultralytics"),
    "DeepFace": ("deepface", "deepface"),
    "MediaPipe": ("mediapipe", "mediapipe"),
}
_OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)


def _opencv_distribution_versions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in _OPENCV_DISTRIBUTIONS:
        try:
            installed[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return installed


def _opencv_runtime_version() -> str:
    import cv2

    return str(cv2.__version__)


def _opencv_installation_error() -> str | None:
    versions = _opencv_distribution_versions()
    providers = set(versions)
    expected_sets = (
        {"opencv-python"},
        {"opencv-python", "opencv-contrib-python"},
    )
    if providers not in expected_sets:
        details = ", ".join(f"{name}={version}" for name, version in sorted(versions.items()))
        return (
            "expected opencv-python alone or a matching regular/contrib pair; "
            f"found {details or 'no OpenCV distribution'}"
        )
    if len(set(versions.values())) != 1:
        details = ", ".join(f"{name}={version}" for name, version in sorted(versions.items()))
        return f"OpenCV distribution versions differ: {details}"
    try:
        runtime = _opencv_runtime_version()
    except (ImportError, AttributeError, RuntimeError) as exc:
        return f"cv2 import failed: {exc}"
    declared = versions["opencv-python"]
    if declared.split(".")[:2] != runtime.split(".")[:2]:
        return f"metadata/runtime mismatch: opencv-python={declared}, cv2={runtime}"
    return None


def _gpu_device_error(device: str) -> str | None:
    """Return why a requested numeric CUDA device cannot be used."""

    if not device.isdecimal():
        return None
    try:
        import torch
    except (ImportError, OSError, RuntimeError) as exc:
        return f"PyTorch could not be loaded: {exc}"
    try:
        if not torch.cuda.is_available():
            return "CUDA is unavailable in this PyTorch installation"
        device_count = int(torch.cuda.device_count())
    except (AttributeError, OSError, RuntimeError) as exc:
        return f"CUDA readiness check failed: {exc}"
    index = int(device)
    if index >= device_count:
        return f"GPU index {index} is out of range; {device_count} CUDA device(s) detected"
    return None


def run_doctor(*, config: DetectionConfig | None = None) -> int:
    config = (config or DetectionConfig()).validated()
    print("AI Detection Suite diagnostics")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    supported = (3, 10) <= sys.version_info[:2] < (3, 14)
    print(f"Supported Python range (3.10-3.13): {'OK' if supported else 'UNSUPPORTED'}")

    required = {"numpy", "OpenCV"}
    if config.enable_objects:
        required.add("Ultralytics")
    if config.enable_face_attributes:
        required.add("DeepFace")
    if config.enable_hands and (
        config.hand_backend == "mediapipe" or config.hand_model is not None
    ):
        required.add("MediaPipe")

    dependencies_ok = supported
    availability: dict[str, bool] = {}
    for label, (module_name, distribution) in PACKAGES.items():
        available = importlib.util.find_spec(module_name) is not None
        availability[label] = available
        try:
            version = metadata.version(distribution) if available else "not installed"
        except metadata.PackageNotFoundError:
            version = "installed, version unknown" if available else "not installed"
        is_required = label in required
        state = "OK" if available else "MISSING" if is_required else "optional"
        print(f"{label}: {version} [{state}]")
        if is_required and not available:
            dependencies_ok = False

    if availability.get("OpenCV", False):
        opencv_error = _opencv_installation_error()
        if opencv_error is None:
            providers = ", ".join(sorted(_opencv_distribution_versions()))
            print(f"OpenCV package set: OK ({providers})")
        else:
            print(f"OpenCV package set: FAILED ({opencv_error})")
            dependencies_ok = False

    if config.enable_objects and availability.get("Ultralytics", False):
        try:
            load_secure_yolo_class()
        except UnsafeUltralyticsError as exc:
            print(f"Restricted YOLO loader: FAILED ({exc})")
            dependencies_ok = False
        else:
            print("Restricted YOLO loader: OK")

        device_error = _gpu_device_error(config.device)
        if device_error is None:
            if config.device.isdecimal():
                print(f"Requested GPU device {config.device}: OK")
        else:
            print(f"Requested GPU device {config.device}: FAILED ({device_error})")
            dependencies_ok = False

    model_ok = True
    if config.enable_objects:
        try:
            model = validate_model_path(
                config.object_model,
                root=config.model_root,
                allow_unverified=config.allow_unverified_model,
            )
        except ModelSecurityError as exc:
            print(f"Model integrity: FAILED ({exc})")
            model_ok = False
        else:
            print(f"Model integrity: OK ({model.name})")
    else:
        print("Model integrity: skipped (object detection disabled)")

    if config.enable_hands and config.hand_model is not None:
        try:
            hand_model = validate_local_asset_path(
                config.hand_model,
                root=config.model_root,
                allowed_suffixes=frozenset({".task"}),
                expected_sha256=config.hand_model_sha256 or "",
                min_size_bytes=1_024,
                max_size_bytes=100 * 1024 * 1024,
            )
        except (ModelSecurityError, OSError, ValueError) as exc:
            print(f"MediaPipe task integrity: FAILED ({exc})")
            model_ok = False
        else:
            print(f"MediaPipe task integrity: OK ({hand_model.name})")

    if config.enable_face_attributes and not config.allow_model_downloads:
        from .detectors import DeepFaceAttributeAnalyzer, _deepface_weights_directory

        if DeepFaceAttributeAnalyzer._offline_weights_available():
            print("DeepFace offline weights: OK")
        else:
            print(
                "DeepFace offline weights: FAILED "
                f"(required files are not provisioned in {_deepface_weights_directory()})"
            )
            model_ok = False

    camera_permission = windows_camera_permission_state()
    if camera_permission == "deny":
        print(
            "Camera permission: BLOCKED (open Windows Settings > Privacy & security > "
            "Camera and allow this Python installation)"
        )
    elif camera_permission == "allow":
        print("Camera permission: allowed by Windows; the camera was not opened")
    else:
        print("Camera was not opened; run the app to test camera permissions.")
    return 0 if dependencies_ok and model_ok else 1
