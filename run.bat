@echo off
title Hide Me Cam
cd /d "%~dp0"
if not exist .venv (
    echo First run: setting up Python packages...
    py -3.13 -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)
.venv\Scripts\python.exe hide_me_cam.py %*
pause
