@echo off
setlocal
rem Drag a dog video, photo or photo folder onto this file (or pass a camera index, e.g. 0).
if "%~1"=="" (
  echo Drag a dog video / photo / folder onto this file, or run: dog-capture.cmd 0  for camera 0
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\dog-capture.ps1" %*
pause
