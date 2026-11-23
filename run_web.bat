@echo off
cd /d "%~dp0"
title Hand Mouse Web Service
echo Starting Hand Mouse Web Controller...
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
echo.
echo Launching Web Dashboard at http://localhost:5000...
echo Keep this terminal window minimized while using the gesture mouse.
echo Press F8 anytime on your laptop to toggle control ON/OFF.
echo Press Ctrl+C in this terminal to shut down the server.
echo.
python web_app.py %*
pause
