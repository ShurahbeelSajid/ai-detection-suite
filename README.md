# All-in-One AI Detection Suite

A local-first webcam application for real-time face, hand, and object detection. The maintained runtime uses bounded background inference, verified local YOLO checkpoints, correct coordinate scaling, and deterministic cleanup.

The default preset provides:

- Haar face detection on CPU.
- A lightweight skin/shape hand-candidate fallback.
- YOLO object detection when the optional Ultralytics dependency is installed.
- An FPS and processing-latency HUD.

Probabilistic age, gender, and expression estimates are deliberately opt-in. No frame, detection, or demographic result is saved by default.

## Requirements

- 64-bit Python 3.10 through 3.13.
- A webcam and desktop session capable of displaying OpenCV windows.
- Windows, Linux, or macOS. The supplied launch scripts target Windows.

Python 3.14 is not currently supported by the full dependency set. A GPU is optional; the suite works on CPU with lower inference rates.

## Quick start on Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\scripts\launch_ai_detection.bat
```

Setup creates an isolated `.venv` with the default face, skin-hand, and YOLO object dependencies; launchers never install packages into the system interpreter. To install only OpenCV/NumPy for face plus skin-hand detection:

```powershell
.\scripts\setup.ps1 -CoreOnly
```

Use `-AllFeatures` only when you also need the optional DeepFace and MediaPipe dependencies:

```powershell
.\scripts\setup.ps1 -AllFeatures
```

Add `-Dev` to any of these profiles to install the test and security-audit tools. Keep the same profile switch when refreshing an existing environment; for example:

```powershell
.\scripts\setup.ps1 -Dev
.\scripts\setup.ps1 -CoreOnly -Dev
.\scripts\setup.ps1 -AllFeatures -Dev
```

Manual setup is also supported:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[objects]"
.\.venv\Scripts\python.exe -m ai_detection_suite --doctor
.\.venv\Scripts\python.exe -m ai_detection_suite
```

For a manual all-features environment, replace `.[objects]` with `.[all]`.

Press `Q` or `Esc` to quit and `R` to reset the FPS average.

## Project layout

- `ai_detection_suite/` - maintained webcam runtime and security boundary.
- `ai_detection_suite/models/` - trusted checkpoints and their integrity manifest.
- `ai_detection_suite/tools/` - installed dataset annotation utilities.
- `scripts/` - Windows setup and launch commands.
- `tests/` - regression and security tests.
- `docs/` - privacy, security, and third-party notices.

## Useful modes

```powershell
# Face detection only
.\.venv\Scripts\python.exe -m ai_detection_suite --no-hands --no-objects

# Verified YOLO object detection only
.\.venv\Scripts\python.exe -m ai_detection_suite --no-faces --no-hands

# Skin/shape hand candidates only
.\.venv\Scripts\python.exe -m ai_detection_suite --no-faces --no-objects --hand-backend skin

# Lower object latency at the cost of more CPU/GPU work
.\.venv\Scripts\python.exe -m ai_detection_suite --object-hz 15 --img-size 320
```

## Face attributes: explicit opt-in

Face attributes are estimates, can be wrong or biased, and must not be used for eligibility, employment, policing, medical, financial, or other high-stakes decisions.

DeepFace requires large attribute weights. The app will not download them silently. An explicit first-run command is required:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[attributes]"
.\.venv\Scripts\python.exe -m ai_detection_suite --no-hands --no-objects --face-attributes --allow-model-downloads
```

Later runs can omit `--allow-model-downloads` after the weights are provisioned. Only confirmed Haar face crops are analyzed, results expire quickly, and results are matched back to current faces by overlap rather than list order. Gender labels and age estimates are displayed without arbitrary overrides or offsets.

See [docs/PRIVACY.md](docs/PRIVACY.md) before enabling this feature.

## MediaPipe hands

Current MediaPipe uses the Tasks API and needs a Hand Landmarker `.task` model. The suite never fetches this asset at runtime. Place a trusted model in a dedicated model directory and pass its digest explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[hands]"
$digest = (Get-FileHash .\custom_models\hand_landmarker.task -Algorithm SHA256).Hash.ToLower()
.\.venv\Scripts\python.exe -m ai_detection_suite `
  --hand-backend mediapipe `
  --model-dir .\custom_models `
  --hand-model hand_landmarker.task `
  --hand-model-sha256 $digest `
  --no-objects
```

Without a task model, `auto` uses the offline skin/shape fallback. The fallback detects candidates, not anatomical landmarks, and its quality varies with lighting and skin tone.

## Model security

PyTorch `.pt` checkpoints use pickle-compatible serialization and must be treated as executable input. Before Ultralytics sees a checkpoint, this project:

- Resolves it relative to an explicit trusted model directory, never the current directory.
- Rejects links, traversal, unsupported extensions, and oversized files.
- Checks exact size and SHA-256 against
  [ai_detection_suite/models/model-manifest.json](ai_detection_suite/models/model-manifest.json).
- Requires PyTorch's restricted-global API and verifies that Ultralytics restricted loading is active.
- Refuses missing files so a model name cannot trigger an automatic download.

The package's `ai_detection_suite/models/` directory is the default model trust root. Deployments can set `AI_DETECTION_MODEL_DIR` or pass `--model-dir`. For a custom checkpoint, add its reviewed hash to `ai_detection_suite/model_security.py`. The `--allow-unverified-model` escape hatch remains available for isolated development, but it weakens file-identity protection; restricted deserialization remains mandatory.

## Why latency is lower

Hand, YOLO, and DeepFace inference each have one long-lived owner worker. While inference is running, at most one newer frame waits; an even newer frame replaces it. This prevents the old thread explosion, concurrent access to one model, stale FIFO backlogs, and unbounded memory growth. A three-failure circuit breaker disables a failing optional backend instead of retrying forever.

Other latency fixes include native-aspect YOLO input, one tensor-to-CPU transfer per result, ROI-only overlays, face-crop attribute analysis, time-based inference rates, source-capture-based result TTLs, native-size display, a one-frame camera buffer request, and processing-scale controls.

## Dataset annotation

The old annotation scripts could overwrite every box except the last one and assigned unstable class IDs. The replacements are deterministic, batched, atomic, hash-check their model, and refuse overwrites unless `--force` is given.

```powershell
# Use model-predicted classes
.\.venv\Scripts\python.exe -m ai_detection_suite.tools.auto_label_yolo --images yolo_dataset\images\train

# Use each filename as its class and YOLO only for box geometry
.\.venv\Scripts\python.exe -m ai_detection_suite.tools.annotate_from_filename --splits train val
```

Input directories must exist and contain supported images. Review generated labels before training; automatic labels are not ground truth.

## Verification

Install the development tools with `-Dev` on the same setup profile first (the default profile uses `.\scripts\setup.ps1 -Dev`).

```powershell
.\.venv\Scripts\python.exe -m compileall -q ai_detection_suite tests
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\bandit.exe -q -r ai_detection_suite
.\.venv\Scripts\pip-audit.exe
```

Unit tests cover latest-frame dropping/single-owner execution, geometry and face association, model path/hash checks, annotation normalization/atomic writes, and import-safe utilities. Webcam and model throughput still depend on the actual machine and should be measured with the live HUD.

## Troubleshooting

Run diagnostics first:

```powershell
.\scripts\launch_ai_detection.bat --doctor
```

If diagnostics report `Camera permission: BLOCKED`, open Windows **Settings >
Privacy & security > Camera** and enable access for Python (or desktop apps).
Setup also detects and repairs conflicting OpenCV installations automatically.

## Privacy and licensing

The previous repository snapshot contained a timestamped demographic inference log. It is not part of the maintained application and should be purged from Git history before publishing an older clone. Runtime data and generated logs are ignored by default.

This project is licensed under `AGPL-3.0-only` to align with the bundled Ultralytics/YOLO components. Commercial/proprietary deployments may require an appropriate Ultralytics Enterprise license and a separate review of every model's terms. See [LICENSE](LICENSE) and [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).

Security reports should follow [docs/SECURITY.md](docs/SECURITY.md).

## Credits

Original project integration by Engr Saad. The demo video from the original project is available on [YouTube](https://youtu.be/tumzfF8CTbI).
