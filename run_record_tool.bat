@echo off
title Video Recorder — Fall Detection Training
echo ============================================
echo   VIDEO RECORDER
echo   Label: TE NGA . NGU/NAM . DI LAI
echo ============================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Tao virtual environment...
    python -m venv venv
    echo.
)

call venv\Scripts\activate.bat

echo [INFO] Video se luu vao: data\recordings\
echo [INFO]   te\   - Te nga
echo [INFO]   ngu\  - Ngu / Nam
echo [INFO]   di\   - Di lai
echo.
echo [RUN] Khoi dong Recorder...
echo.

python data\record_tool.py

pause
