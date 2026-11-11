@echo off
setlocal
if "%~1"=="" (
  echo Drag a dog video onto this file to select front, left and right views.
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run-dog-capture.ps1" -Video "%~1" -Mode views
pause
