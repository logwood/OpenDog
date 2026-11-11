@echo off
if "%~1"=="" (
  echo Drag a dog video onto this file for real-time replay.
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run-dog-capture.ps1" -Video "%~1" -Realtime -Show
pause
