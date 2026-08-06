"""Side-effect-free operating-system checks used for actionable diagnostics."""

from __future__ import annotations

import os
import sys
from pathlib import PureWindowsPath


def _store_package_family(base_prefix: str) -> str | None:
    """Return the Microsoft Store package family encoded in a Python path."""

    name = PureWindowsPath(base_prefix).name
    parts = name.split("_")
    product = parts[0] if parts else ""
    publisher = parts[-1] if len(parts) >= 5 else ""
    if product.startswith("PythonSoftwareFoundation.Python.") and publisher:
        return f"{product}_{publisher}"
    return None


def _camera_consent_value(subkey: str) -> str | None:
    try:
        import winreg
    except ImportError:  # pragma: no cover - Windows-only module
        return None

    base = (
        r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager"
        r"\ConsentStore\webcam"
    )
    path = f"{base}\\{subkey}" if subkey else base
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
            value, _kind = winreg.QueryValueEx(key, "Value")
    except OSError:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in {"allow", "deny", "prompt"} else None


def windows_camera_permission_state() -> str | None:
    """Return the effective Windows camera consent value when discoverable."""

    if os.name != "nt":
        return None
    family = _store_package_family(sys.base_prefix)
    candidates = [family] if family else ["NonPackaged"]
    candidates.append("")
    for candidate in candidates:
        if candidate is None:
            continue
        state = _camera_consent_value(candidate)
        if state is not None:
            return state
    return None


def camera_permission_hint() -> str | None:
    """Return actionable guidance only when Windows blocks camera access."""

    if windows_camera_permission_state() != "deny":
        return None
    return (
        "Windows camera access is denied for this Python installation. Open "
        "Settings > Privacy & security > Camera, then enable Camera access and "
        "the Python entry (or 'Let desktop apps access your camera')."
    )


__all__ = ["camera_permission_hint", "windows_camera_permission_state"]
