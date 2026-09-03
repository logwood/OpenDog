[CmdletBinding()]
param(
    [switch]$AcceptAndroidSdkLicense
)

$ErrorActionPreference = "Stop"

if (-not $AcceptAndroidSdkLicense) {
    throw "Pass -AcceptAndroidSdkLicense explicitly to confirm that you are authorized to accept the Android SDK License Agreement."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$tempRoot = Join-Path $projectRoot ".tmp"
$bootstrapRoot = Join-Path $tempRoot "android-bootstrap"
$sdkRoot = Join-Path $tempRoot "android-sdk"
$distributionRoot = Join-Path $tempRoot "gradle-distributions"
$gradleHome = Join-Path $tempRoot "gradle-home"
$javaHome = "C:\Program Files\Microsoft\jdk-21.0.6.7-hotspot"

$commandLineVersion = "15859902"
$commandLineArchive = "commandlinetools-win-$($commandLineVersion)_latest.zip"
$commandLineUrl = "https://dl.google.com/android/repository/$commandLineArchive"
$commandLineSha256 = "90ae805d20434428bffcb699c290860f19bb5f66a67e6b330067e3de801fb04a"
$gradleVersion = "9.4.1"
$gradleArchive = "gradle-$gradleVersion-bin.zip"
$gradleUrl = "https://services.gradle.org/distributions/$gradleArchive"
$gradleShaUrl = "$gradleUrl.sha256"

function Assert-WorkspacePath([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    $prefix = $projectRoot.TrimEnd("\") + "\"
    if (-not $full.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside the workspace: $full"
    }
}

function Invoke-CurlDownload([string]$Url, [string]$Destination) {
    Assert-WorkspacePath $Destination
    & curl.exe -L --fail --retry 3 --connect-timeout 30 -o $Destination $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed with exit code ${LASTEXITCODE}: $Url"
    }
}

function Get-Sha256Hex([string]$Path) {
    $fileHashCommand = Get-Command Get-FileHash -ErrorAction SilentlyContinue
    if ($fileHashCommand) {
        return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $lines = & certutil.exe -hashfile $Path SHA256 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Neither Get-FileHash nor certutil could calculate a SHA-256 hash."
    }
    $hashLine = $lines |
        ForEach-Object { $_.ToString().Trim() } |
        Where-Object { $_ -match "^[0-9A-Fa-f]{64}$" } |
        Select-Object -First 1
    if (-not $hashLine) {
        throw "certutil returned no SHA-256 hash for $Path."
    }
    return $hashLine.ToLowerInvariant()
}

function Assert-FileHash([string]$Path, [string]$Expected) {
    $actual = Get-Sha256Hex $Path
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path. Expected $Expected, got $actual."
    }
}

foreach ($path in @($bootstrapRoot, $sdkRoot, $distributionRoot, $gradleHome)) {
    Assert-WorkspacePath $path
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}
if (-not (Test-Path -LiteralPath $javaHome -PathType Container)) {
    throw "The confirmed JDK was not found: $javaHome"
}

$commandLineZip = Join-Path $bootstrapRoot $commandLineArchive
if (-not (Test-Path -LiteralPath $commandLineZip)) {
    Invoke-CurlDownload $commandLineUrl $commandLineZip
}
Assert-FileHash $commandLineZip $commandLineSha256

$sdkManager = Join-Path $sdkRoot "cmdline-tools\latest\bin\sdkmanager.bat"
if (-not (Test-Path -LiteralPath $sdkManager)) {
    $commandLineExtract = Join-Path $bootstrapRoot "cmdline-tools-$commandLineVersion"
    Assert-WorkspacePath $commandLineExtract
    New-Item -ItemType Directory -Path $commandLineExtract -Force | Out-Null
    Expand-Archive -LiteralPath $commandLineZip -DestinationPath $commandLineExtract -Force
    $extractedTools = Join-Path $commandLineExtract "cmdline-tools"
    $latestTools = Join-Path $sdkRoot "cmdline-tools\latest"
    New-Item -ItemType Directory -Path $latestTools -Force | Out-Null
    Get-ChildItem -LiteralPath $extractedTools -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $latestTools -Recurse -Force
    }
}
if (-not (Test-Path -LiteralPath $sdkManager)) {
    throw "sdkmanager was not found after extracting Android command-line tools: $sdkManager"
}

$env:JAVA_HOME = $javaHome
$env:ANDROID_HOME = $sdkRoot
$env:ANDROID_SDK_ROOT = $sdkRoot
$env:GRADLE_USER_HOME = $gradleHome

1..100 | ForEach-Object { "y" } | & $sdkManager --sdk_root=$sdkRoot --licenses
if ($LASTEXITCODE -ne 0) {
    throw "Android SDK license processing failed with exit code $LASTEXITCODE."
}

& $sdkManager --sdk_root=$sdkRoot "platform-tools" "platforms;android-36" "build-tools;36.0.0"
if ($LASTEXITCODE -ne 0) {
    throw "Android SDK component installation failed with exit code $LASTEXITCODE."
}

$gradleZip = Join-Path $bootstrapRoot $gradleArchive
$gradleShaFile = Join-Path $bootstrapRoot "$gradleArchive.sha256"
if (-not (Test-Path -LiteralPath $gradleShaFile)) {
    Invoke-CurlDownload $gradleShaUrl $gradleShaFile
}
$gradleSha256 = ((Get-Content -LiteralPath $gradleShaFile -Raw).Trim() -split "\s+")[0]
if (-not (Test-Path -LiteralPath $gradleZip)) {
    Invoke-CurlDownload $gradleUrl $gradleZip
}
Assert-FileHash $gradleZip $gradleSha256

$gradleDirectory = Join-Path $distributionRoot "gradle-$gradleVersion"
if (-not (Test-Path -LiteralPath (Join-Path $gradleDirectory "bin\gradle.bat"))) {
    Expand-Archive -LiteralPath $gradleZip -DestinationPath $distributionRoot -Force
}
$gradle = Join-Path $gradleDirectory "bin\gradle.bat"
if (-not (Test-Path -LiteralPath $gradle)) {
    throw "Gradle was not found after extraction: $gradle"
}

Push-Location $PSScriptRoot
try {
    & $gradle wrapper --gradle-version $gradleVersion --distribution-type bin --no-daemon
    if ($LASTEXITCODE -ne 0) {
        throw "Gradle wrapper generation failed with exit code $LASTEXITCODE."
    }
    $wrapperProperties = Join-Path $PSScriptRoot "gradle\wrapper\gradle-wrapper.properties"
    $wrapperText = Get-Content -LiteralPath $wrapperProperties -Raw
    $wrapperText = $wrapperText -replace "networkTimeout=\d+", "networkTimeout=60000"
    if ($wrapperText -notmatch "(?m)^distributionSha256Sum=") {
        $wrapperText = $wrapperText.TrimEnd() + [Environment]::NewLine + "distributionSha256Sum=$gradleSha256"
    }
    Set-Content -LiteralPath $wrapperProperties -Value $wrapperText.TrimEnd() -Encoding ASCII
} finally {
    Pop-Location
}

Write-Output "ANDROID_SDK_ROOT=$sdkRoot"
Write-Output "GRADLE=$gradle"
Write-Output "Toolchain bootstrap completed."
