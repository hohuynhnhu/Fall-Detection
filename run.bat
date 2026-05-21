@echo off
title Fall Detection — Desktop App
echo ============================================
echo   FALL DETECTION DESKTOP APP
echo   MediaPipe + YOLO Pose
echo ============================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Tao virtual environment...
    python -m venv venv
    echo.
)

call venv\Scripts\activate.bat

echo [SETUP] Kiem tra / cai packages...
pip install -r requirements.txt -q
echo.

echo [RUN] Khoi dong desktop app...
echo [INFO] Backend URL: http://localhost:8000 (chay run.bat trong fall-detection-backend truoc)
echo.

cd src
python app.py

pause
