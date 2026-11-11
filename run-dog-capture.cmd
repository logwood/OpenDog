@echo off
if "%~1"=="" (
  echo Drag a dog video onto this file, or run: run-dog-capture.cmd "C:\path\dog.mp4"
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run-dog-capture.ps1" -Video "%~1"
pause
