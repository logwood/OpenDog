@echo off
setlocal
set "ANDROID_PROJECT=%~dp0src\Pet-ReID-IMAG\frontend\pet-reid-android"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ANDROID_PROJECT%\build-apk.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo Android APK build failed with exit code %EXIT_CODE%.
)
exit /b %EXIT_CODE%
