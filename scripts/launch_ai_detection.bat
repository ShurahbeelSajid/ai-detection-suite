@echo off
setlocal
pushd "%~dp0.." || exit /b 1

set "PROJECT_ROOT=%CD%"
set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Virtual environment not found.
    echo Run: powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\setup.ps1"
    popd
    exit /b 2
)

"%PYTHON_EXE%" -m ai_detection_suite %*
set "APP_EXIT=%ERRORLEVEL%"
popd
exit /b %APP_EXIT%
