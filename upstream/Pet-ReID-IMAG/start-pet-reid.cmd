@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\pet-reid-stack.ps1" start -Provider cuda
if errorlevel 1 (
  echo.
  echo Pet ReID failed to start. See logs\quick_start for details.
  pause
)
endlocal
