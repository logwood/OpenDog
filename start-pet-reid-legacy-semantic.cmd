@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\pet-reid-stack.ps1" start -Provider cuda -Model legacy-semantic
if errorlevel 1 (
  echo.
  echo Legacy semantic compatibility mode failed to start. See artifacts\workspace_logs\quick_start for details.
  pause
)
endlocal
