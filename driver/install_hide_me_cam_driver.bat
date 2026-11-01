@echo off
:: Installs the Unity Capture virtual camera driver under the name "Hide Me Cam".
:: Only needed if you don't have OBS Studio.
::
:: 1. Download https://github.com/schellingb/UnityCapture/archive/refs/heads/master.zip and unzip it
:: 2. Copy this file into the unzipped "Install" folder (next to UnityCaptureFilter64.dll)
:: 3. Double-click it and allow the admin prompt

net session >nul 2>&1
if errorlevel 1 (
    echo Requesting administrator rights...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

cd /d "%~dp0"
if not exist "UnityCaptureFilter64.dll" (
    echo UnityCaptureFilter64.dll not found. Put this file in Unity Capture's "Install" folder.
    pause
    exit /b 1
)

regsvr32 /s "UnityCaptureFilter32.dll" "/i:UnityCaptureName=Hide Me Cam"
regsvr32 /s "UnityCaptureFilter64.dll" "/i:UnityCaptureName=Hide Me Cam"
echo Done. "Hide Me Cam" should now appear as a camera in your apps.
echo To remove it later, run Uninstall.bat from the same folder.
pause
