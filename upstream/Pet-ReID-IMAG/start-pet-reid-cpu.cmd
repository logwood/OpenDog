@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\pet-reid-stack.ps1" start -Provider cpu
if errorlevel 1 (
  echo.
  echo Pet ReID CPU mode failed to start. See logs\quick_start for details.
  pause
)
endlocal
