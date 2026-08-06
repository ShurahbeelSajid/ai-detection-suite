"""Command-line interface for the maintained detection application."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .config import DetectionConfig, preset_config


def build_parser(*, preset: str = "full") -> argparse.ArgumentParser:
    defaults = preset_config(preset)
    parser = argparse.ArgumentParser(
        description="Low-latency local webcam detection with bounded inference workers.",
    )
    parser.add_argument("--camera", type=int, default=defaults.camera_index, help="camera index")
    parser.add_argument("--width", type=int, default=defaults.capture_width)
    parser.add_argument("--height", type=int, default=defaults.capture_height)
    parser.add_argument("--fps", type=int, default=defaults.capture_fps)
    parser.add_argument("--processing-scale", type=float, default=defaults.processing_scale)
    parser.add_argument("--face-stride", type=int, default=defaults.face_stride)
    parser.add_argument("--hand-stride", type=int, default=defaults.hand_stride)
    parser.add_argument("--hand-ttl", type=float, default=defaults.hand_ttl)
    parser.add_argument("--object-hz", type=float, default=defaults.object_hz)
    parser.add_argument("--attribute-hz", type=float, default=defaults.attribute_hz)
    parser.add_argument("--object-ttl", type=float, default=defaults.object_ttl)
    parser.add_argument("--attribute-ttl", type=float, default=defaults.attribute_ttl)
    parser.add_argument(
        "--model", default=defaults.object_model, help="verified local YOLO .pt file"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=defaults.model_root,
        help="trusted directory containing local model assets",
    )
    parser.add_argument("--img-size", type=int, default=defaults.object_img_size)
    parser.add_argument("--confidence", type=float, default=defaults.object_confidence)
    parser.add_argument("--iou", type=float, default=defaults.object_iou)
    parser.add_argument("--max-detections", type=int, default=defaults.object_max_detections)
    parser.add_argument("--device", default=defaults.device, help="auto, cpu, or GPU index")
    parser.add_argument(
        "--hand-backend",
        choices=("auto", "skin", "mediapipe"),
        default=defaults.hand_backend,
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=defaults.hand_model,
        help="pre-provisioned MediaPipe Hand Landmarker .task file",
    )
    parser.add_argument(
        "--hand-model-sha256",
        default=defaults.hand_model_sha256,
        help="expected SHA-256 of the MediaPipe .task file",
    )
    parser.add_argument(
        "--faces",
        action=argparse.BooleanOptionalAction,
        default=defaults.enable_faces,
    )
    parser.add_argument(
        "--hands",
        action=argparse.BooleanOptionalAction,
        default=defaults.enable_hands,
    )
    parser.add_argument(
        "--objects",
        action=argparse.BooleanOptionalAction,
        default=defaults.enable_objects,
    )
    parser.add_argument(
        "--face-attributes",
        action=argparse.BooleanOptionalAction,
        default=defaults.enable_face_attributes,
        help="opt in to probabilistic age/gender/expression analysis",
    )
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=defaults.mirror,
    )
    parser.add_argument(
        "--allow-unverified-model",
        action="store_true",
        default=defaults.allow_unverified_model,
        help="dangerous: allow a custom local checkpoint after path/size checks",
    )
    parser.add_argument(
        "--allow-model-downloads",
        action="store_true",
        default=defaults.allow_model_downloads,
        help="allow explicitly requested DeepFace to download missing upstream weights",
    )
    parser.add_argument(
        "--strict-features",
        action="store_true",
        default=defaults.strict_features,
        help="exit instead of gracefully disabling a missing optional feature",
    )
    parser.add_argument(
        "--max-frames", type=int, default=defaults.max_frames, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--doctor", action="store_true", help="check installation without opening camera"
    )
    return parser


def parse_config(
    argv: Sequence[str] | None = None,
    *,
    preset: str = "full",
) -> tuple[DetectionConfig, bool]:
    args = build_parser(preset=preset).parse_args(argv)
    base = preset_config(preset)
    config = replace(
        base,
        camera_index=args.camera,
        capture_width=args.width,
        capture_height=args.height,
        capture_fps=args.fps,
        processing_scale=args.processing_scale,
        face_stride=args.face_stride,
        hand_stride=args.hand_stride,
        hand_ttl=args.hand_ttl,
        object_hz=args.object_hz,
        attribute_hz=args.attribute_hz,
        object_ttl=args.object_ttl,
        attribute_ttl=args.attribute_ttl,
        object_model=args.model,
        model_root=args.model_dir,
        object_img_size=args.img_size,
        object_confidence=args.confidence,
        object_iou=args.iou,
        object_max_detections=args.max_detections,
        device=args.device,
        hand_backend=args.hand_backend,
        hand_model=args.hand_model,
        hand_model_sha256=args.hand_model_sha256,
        enable_faces=args.faces,
        enable_hands=args.hands,
        enable_objects=args.objects,
        enable_face_attributes=args.face_attributes,
        allow_unverified_model=args.allow_unverified_model,
        allow_model_downloads=args.allow_model_downloads,
        strict_features=args.strict_features,
        mirror=args.mirror,
        max_frames=args.max_frames,
    ).validated()
    return config, bool(args.doctor)


def run_cli(argv: Sequence[str] | None = None, *, preset: str = "full") -> int:
    try:
        config, doctor = parse_config(argv, preset=preset)
        if doctor:
            from .diagnostics import run_doctor

            return run_doctor(config=config)
        if config.enable_face_attributes:
            print(
                "NOTICE: face attributes are probabilistic and sensitive. "
                "They are not logged and must not be used for high-stakes decisions."
            )
        from .app import run_app

        return run_app(config)
    except KeyboardInterrupt:
        return 130
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
