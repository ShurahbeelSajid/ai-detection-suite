"""Fail-closed validation for local PyTorch model checkpoints.

Call :func:`validate_model_path` before passing a path to ``ultralytics.YOLO``
or any other deserializer. Unless ``AI_DETECTION_MODEL_DIR`` is set, the default
trust root is the package's dedicated models directory, derived from this module
rather than the process working directory.

Hash allowlisting protects against accidental or malicious model replacement
only while this module and its allowlist remain trusted.  It does not make
pickle deserialization intrinsically safe and cannot prevent a privileged
attacker from replacing a file after validation.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .config import DEFAULT_MODEL_ROOT, default_model_root

PathInput = str | os.PathLike[str]

ALLOWED_MODEL_SUFFIXES = frozenset({".pt"})
MAX_MODEL_SIZE_BYTES = 256 * 1024 * 1024
_HASH_CHUNK_SIZE = 1024 * 1024


class ModelSecurityError(ValueError):
    """Base exception for rejected model paths or model contents."""


class ModelPathError(ModelSecurityError):
    """Raised when a model path escapes the trust root or is not a regular file."""


class UnverifiedModelError(ModelSecurityError):
    """Raised when a model has no allowlist entry and no explicit override."""


class ModelIntegrityError(ModelSecurityError):
    """Raised when an allowlisted model's size or digest does not match."""


@dataclass(frozen=True)
class ModelAllowlistEntry:
    """Expected immutable identity of one model artifact."""

    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        digest = self.sha256.lower()
        if self.size_bytes <= 0:
            raise ValueError("allowlisted model size must be positive")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("allowlisted model SHA-256 must be 64 hexadecimal characters")
        object.__setattr__(self, "sha256", digest)


DEFAULT_MODEL_ALLOWLIST: Mapping[str, ModelAllowlistEntry] = MappingProxyType(
    {
        "yolo11n.pt": ModelAllowlistEntry(
            size_bytes=5_613_764,
            sha256="0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        ),
    }
)


def sha256_file(path: PathInput) -> str:
    """Return a lowercase SHA-256 digest without deserializing the file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_link_like(path: Path) -> bool:
    """Recognize symbolic links and Windows directory junctions."""

    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _reject_link_components(candidate: Path, root: Path) -> None:
    current = root
    for component in candidate.relative_to(root).parts:
        current = current / component
        if _is_link_like(current):
            raise ModelPathError(f"model path must not traverse a link or junction: {current}")


def _file_identity(file_stat: os.stat_result) -> tuple[int, int, int, int]:
    """Fields sufficient to detect common replacement during hashing."""

    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )


def validate_model_path(
    model_path: PathInput,
    *,
    root: PathInput | None = None,
    allowlist: Mapping[str, ModelAllowlistEntry] | None = None,
    allow_unverified: bool = False,
    max_size_bytes: int = MAX_MODEL_SIZE_BYTES,
) -> Path:
    """Validate and return an absolute model path without loading the model.

    Relative paths are anchored to ``root``. Without an explicit root, the
    ``AI_DETECTION_MODEL_DIR`` override or ``DEFAULT_MODEL_ROOT`` is used, never
    the current working directory. Allowlist keys are root-relative POSIX paths,
    such as ``"yolo11n.pt"``.

    ``allow_unverified=True`` is an explicit escape hatch for custom models. It
    permits a model with no allowlist entry after all path, suffix, regular-file,
    and size checks pass. It deliberately does not bypass a mismatch for a known
    allowlisted path, because that is a strong signal of replacement or damage.
    """

    if max_size_bytes <= 0:
        raise ValueError("max_size_bytes must be positive")

    root_input = Path(root).expanduser() if root is not None else default_model_root()
    if _is_link_like(root_input):
        raise ModelPathError(f"model trust root must not be a link or junction: {root_input}")
    try:
        trusted_root = root_input.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ModelPathError(f"model trust root does not exist: {root_input}") from exc
    if not trusted_root.is_dir():
        raise ModelPathError(f"model trust root is not a directory: {trusted_root}")

    supplied_path = Path(model_path).expanduser()
    candidate_input = supplied_path if supplied_path.is_absolute() else trusted_root / supplied_path
    candidate_lexical = Path(os.path.abspath(os.fspath(candidate_input)))

    if not _is_within(candidate_lexical, trusted_root):
        raise ModelPathError(f"model path is outside the trusted root: {candidate_lexical}")
    _reject_link_components(candidate_lexical, trusted_root)

    try:
        candidate = candidate_lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ModelPathError(f"model file does not exist: {candidate_lexical}") from exc
    if candidate != candidate_lexical or not _is_within(candidate, trusted_root):
        raise ModelPathError(
            f"model path resolves outside the trusted root or through a link: {candidate}"
        )
    if candidate.suffix.lower() not in ALLOWED_MODEL_SUFFIXES:
        raise ModelPathError(
            f"unsupported model suffix {candidate.suffix!r}; expected one of "
            f"{sorted(ALLOWED_MODEL_SUFFIXES)}"
        )

    file_lstat = candidate.lstat()
    if not stat.S_ISREG(file_lstat.st_mode):
        raise ModelPathError(f"model path is not a regular file: {candidate}")
    if file_lstat.st_size <= 0:
        raise ModelPathError(f"model file is empty: {candidate}")
    if file_lstat.st_size > max_size_bytes:
        raise ModelPathError(
            f"model file is too large ({file_lstat.st_size} bytes; limit is {max_size_bytes}): {candidate}"
        )

    effective_allowlist = DEFAULT_MODEL_ALLOWLIST if allowlist is None else allowlist
    relative_name = candidate.relative_to(trusted_root).as_posix()
    expected = effective_allowlist.get(relative_name)
    if expected is None:
        if allow_unverified:
            return candidate
        raise UnverifiedModelError(
            f"model is not allowlisted: {relative_name}; use allow_unverified=True only for an explicitly trusted custom model"
        )
    if not isinstance(expected, ModelAllowlistEntry):
        raise TypeError(f"allowlist entry for {relative_name!r} must be a ModelAllowlistEntry")
    if file_lstat.st_size != expected.size_bytes:
        raise ModelIntegrityError(
            f"model size mismatch for {relative_name}: expected {expected.size_bytes}, got {file_lstat.st_size}"
        )

    identity_before = _file_identity(candidate.stat())
    actual_digest = sha256_file(candidate)
    identity_after = _file_identity(candidate.stat())
    if identity_before != identity_after:
        raise ModelIntegrityError(f"model changed while it was being verified: {relative_name}")
    if not hmac.compare_digest(actual_digest, expected.sha256):
        raise ModelIntegrityError(f"model SHA-256 mismatch for {relative_name}")

    return candidate


def validate_local_asset_path(
    asset_path: PathInput,
    *,
    root: PathInput,
    allowed_suffixes: frozenset[str],
    expected_sha256: str,
    min_size_bytes: int = 1,
    max_size_bytes: int = MAX_MODEL_SIZE_BYTES,
) -> Path:
    """Validate a user-provisioned binary asset under an explicit trust root."""

    normalized_suffixes = frozenset(suffix.lower() for suffix in allowed_suffixes)
    if not normalized_suffixes or any(not suffix.startswith(".") for suffix in normalized_suffixes):
        raise ValueError("allowed_suffixes must contain dotted file suffixes")
    if min_size_bytes < 1 or max_size_bytes < min_size_bytes:
        raise ValueError("invalid asset size bounds")
    digest = expected_sha256.lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("expected SHA-256 must contain 64 hexadecimal characters")

    root_input = Path(root).expanduser()
    if _is_link_like(root_input):
        raise ModelPathError(f"asset trust root must not be a link or junction: {root_input}")
    try:
        trusted_root = root_input.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ModelPathError(f"asset trust root does not exist: {root_input}") from exc
    if not trusted_root.is_dir():
        raise ModelPathError(f"asset trust root is not a directory: {trusted_root}")

    supplied = Path(asset_path).expanduser()
    candidate_input = supplied if supplied.is_absolute() else trusted_root / supplied
    candidate_lexical = Path(os.path.abspath(os.fspath(candidate_input)))
    if not _is_within(candidate_lexical, trusted_root):
        raise ModelPathError(f"asset path is outside the trusted root: {candidate_lexical}")
    _reject_link_components(candidate_lexical, trusted_root)
    try:
        candidate = candidate_lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ModelPathError(f"asset file does not exist: {candidate_lexical}") from exc
    if candidate != candidate_lexical or not _is_within(candidate, trusted_root):
        raise ModelPathError(f"asset path resolves outside the trust root: {candidate}")
    if candidate.suffix.lower() not in normalized_suffixes:
        raise ModelPathError(
            f"unsupported asset suffix {candidate.suffix!r}; expected one of {sorted(normalized_suffixes)}"
        )

    file_lstat = candidate.lstat()
    if not stat.S_ISREG(file_lstat.st_mode):
        raise ModelPathError(f"asset path is not a regular file: {candidate}")
    if not min_size_bytes <= file_lstat.st_size <= max_size_bytes:
        raise ModelPathError(
            f"asset size {file_lstat.st_size} is outside "
            f"[{min_size_bytes}, {max_size_bytes}] bytes: {candidate}"
        )

    identity_before = _file_identity(candidate.stat())
    actual_digest = sha256_file(candidate)
    identity_after = _file_identity(candidate.stat())
    if identity_before != identity_after:
        raise ModelIntegrityError(f"asset changed while it was being verified: {candidate.name}")
    if not hmac.compare_digest(actual_digest, digest):
        raise ModelIntegrityError(f"asset SHA-256 mismatch for {candidate.name}")
    return candidate


__all__ = [
    "ALLOWED_MODEL_SUFFIXES",
    "DEFAULT_MODEL_ALLOWLIST",
    "DEFAULT_MODEL_ROOT",
    "MAX_MODEL_SIZE_BYTES",
    "ModelAllowlistEntry",
    "ModelIntegrityError",
    "ModelPathError",
    "ModelSecurityError",
    "UnverifiedModelError",
    "sha256_file",
    "validate_local_asset_path",
    "validate_model_path",
]
