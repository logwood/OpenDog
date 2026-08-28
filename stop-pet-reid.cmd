@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\pet-reid-stack.ps1" stop -NoBrowser
if errorlevel 1 pause
endlocal
