@echo off
if "%~1"=="" (
  echo Drag a video onto this file.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run-dog-trt.ps1" -Video "%~1" -Realtime -Show
if errorlevel 1 pause
