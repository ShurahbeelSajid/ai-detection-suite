param(
    [switch]$CoreOnly,
    [switch]$AllFeatures,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
if ($CoreOnly -and $AllFeatures) {
    throw "Choose either -CoreOnly or -AllFeatures, not both."
}
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$reusingVenv = Test-Path -LiteralPath $venvPython

Push-Location $projectRoot
try {
    if (-not $reusingVenv) {
        $created = $false
        foreach ($version in @("3.13", "3.12", "3.11", "3.10")) {
            & py "-$version" -m venv .venv 2>$null
            if ($LASTEXITCODE -eq 0) {
                $created = $true
                break
            }
        }
        if (-not $created) {
            throw "Install a supported 64-bit Python (3.10 through 3.13) first."
        }
    }

    & $venvPython -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "The existing .venv uses an unsupported Python. Recreate it with Python 3.10 through 3.13."
    }

    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "pip upgrade failed."
    }

    if ($reusingVenv) {
        $excludedPackages = @()
        if ($CoreOnly) {
            $excludedPackages += @(
                "ultralytics",
                "ultralytics-thop",
                "torch",
                "torchvision",
                "deepface",
                "tf-keras",
                "mediapipe"
            )
        }
        elseif (-not $AllFeatures) {
            $excludedPackages += @("deepface", "tf-keras", "mediapipe")
        }
        if (-not $Dev) {
            $excludedPackages += @("bandit", "pip-audit", "pytest", "ruff")
        }
        $excludedPackages = @($excludedPackages | Sort-Object -Unique)
        if ($excludedPackages.Count -gt 0) {
            Write-Host "Removing packages excluded by the selected setup mode: $($excludedPackages -join ', ')"
            & $venvPython -m pip uninstall --yes $excludedPackages
            if ($LASTEXITCODE -ne 0) {
                throw "Could not remove packages excluded by the selected setup mode."
            }
        }
    }

    $selectedExtras = @()
    if (-not $CoreOnly) {
        $selectedExtras += $(if ($AllFeatures) { "all" } else { "objects" })
    }
    if ($Dev) {
        $selectedExtras += "dev"
    }
    $installTarget = if ($selectedExtras.Count -gt 0) { ".[$($selectedExtras -join ',')]" } else { "." }
    & $venvPython -m pip install -e $installTarget
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }

    $openCvProfile = if ($AllFeatures) { "all" } else { "core" }
    $openCvState = & $venvPython -c "import cv2, sys; from importlib.metadata import packages_distributions, version; expected={'opencv-python', 'opencv-contrib-python'} if sys.argv[1] == 'all' else {'opencv-python'}; providers=set(packages_distributions().get('cv2', ())); versions={name: version(name) for name in providers}; valid=providers == expected and len(set(versions.values())) == 1 and version('opencv-python').split('.')[:2] == cv2.__version__.split('.')[:2]; print('OK' if valid else f'providers={sorted(providers)}, versions={versions}, runtime={cv2.__version__}')" $openCvProfile
    if ($LASTEXITCODE -ne 0 -or $openCvState -ne "OK") {
        Write-Host "Repairing the OpenCV package set: $openCvState"
        & $venvPython -m pip uninstall --yes opencv-contrib-python opencv-contrib-python-headless opencv-python opencv-python-headless
        if ($LASTEXITCODE -ne 0) {
            throw "Could not remove conflicting OpenCV packages."
        }
        $openCvPackages = @("opencv-python==4.14.0.94")
        if ($AllFeatures) {
            $openCvPackages += "opencv-contrib-python==4.14.0.94"
        }
        & $venvPython -m pip install --force-reinstall --no-deps $openCvPackages
        if ($LASTEXITCODE -ne 0) {
            throw "Could not restore the supported OpenCV package."
        }
    }

    & $venvPython -c "import cv2, sys; from importlib.metadata import packages_distributions, version; providers=set(packages_distributions().get('cv2', ())); expected={'opencv-python', 'opencv-contrib-python'} if sys.argv[1] == 'all' else {'opencv-python'}; assert providers == expected, f'unexpected cv2 providers: {sorted(providers)}'; versions={name: version(name) for name in providers}; assert len(set(versions.values())) == 1, f'OpenCV distribution versions differ: {versions}'; declared=version('opencv-python'); runtime=cv2.__version__; assert declared.split('.')[:2] == runtime.split('.')[:2], f'OpenCV metadata/runtime mismatch: {declared} != {runtime}'; print(f'OpenCV runtime: {runtime}; providers: {sorted(providers)} [OK]')" $openCvProfile
    if ($LASTEXITCODE -ne 0) {
        throw "OpenCV validation failed."
    }

    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "Installed dependencies are inconsistent."
    }

    if ($AllFeatures) {
        $tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
        $smokeHome = Join-Path $tempRoot ("ai-detection-deepface-" + [System.Guid]::NewGuid().ToString("N"))
        $resolvedSmokeHome = [System.IO.Path]::GetFullPath($smokeHome)
        if (-not $resolvedSmokeHome.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to create an import-smoke directory outside the system temporary directory."
        }

        New-Item -ItemType Directory -Path $resolvedSmokeHome -ErrorAction Stop | Out-Null
        $previousDeepFaceHome = $env:DEEPFACE_HOME
        $previousTfLogLevel = $env:TF_CPP_MIN_LOG_LEVEL
        try {
            $env:DEEPFACE_HOME = $resolvedSmokeHome
            $env:TF_CPP_MIN_LOG_LEVEL = "3"
            & $venvPython -c "from deepface import DeepFace; import mediapipe as mp; from mediapipe.tasks import python as mp_python; from mediapipe.tasks.python import vision; assert callable(DeepFace.analyze); assert mp.Image is not None; assert mp_python.BaseOptions is not None; assert vision.HandLandmarker is not None; print('DeepFace/MediaPipe imports: OK')"
            if ($LASTEXITCODE -ne 0) {
                throw "All-features import smoke test failed."
            }
        }
        finally {
            if ($null -eq $previousDeepFaceHome) {
                Remove-Item Env:\DEEPFACE_HOME -ErrorAction SilentlyContinue
            }
            else {
                $env:DEEPFACE_HOME = $previousDeepFaceHome
            }
            if ($null -eq $previousTfLogLevel) {
                Remove-Item Env:\TF_CPP_MIN_LOG_LEVEL -ErrorAction SilentlyContinue
            }
            else {
                $env:TF_CPP_MIN_LOG_LEVEL = $previousTfLogLevel
            }
            if (Test-Path -LiteralPath $resolvedSmokeHome) {
                Remove-Item -LiteralPath $resolvedSmokeHome -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    if ($CoreOnly) {
        & $venvPython -m ai_detection_suite --doctor --no-objects
    }
    else {
        & $venvPython -m ai_detection_suite --doctor
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Installation diagnostics failed."
    }
}
finally {
    Pop-Location
}
