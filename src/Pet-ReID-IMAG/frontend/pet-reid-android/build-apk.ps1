[CmdletBinding()]
param(
    [switch]$SkipCopy
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$sdkRoot = Join-Path $projectRoot ".tmp\android-sdk"
$gradleHome = Join-Path $projectRoot ".tmp\gradle-home"
$artifactRoot = Join-Path $projectRoot "artifacts\releases\pawprint-id-android"
$jdkCandidates = @(
    $env:JAVA_HOME,
    "C:\Program Files\Microsoft\jdk-21.0.6.7-hotspot",
    "C:\Program Files\Eclipse Adoptium\jdk-21*",
    "C:\Program Files\Java\jdk-17*"
)

function Resolve-ExistingDirectory([string[]]$Candidates) {
    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
        $resolved = Get-Item -Path $candidate -ErrorAction SilentlyContinue |
            Where-Object { $_.PSIsContainer } |
            Select-Object -First 1
        if ($resolved) {
            return $resolved.FullName
        }
    }
    return $null
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

$javaHome = Resolve-ExistingDirectory $jdkCandidates
if (-not $javaHome) {
    throw "JDK not found. Set JAVA_HOME or install JDK 17+."
}
$sdkManager = Join-Path $sdkRoot "cmdline-tools\latest\bin\sdkmanager.bat"
$buildTools = Join-Path $sdkRoot "build-tools\36.0.0"
$platform = Join-Path $sdkRoot "platforms\android-36"
if (-not (Test-Path -LiteralPath $sdkManager) -or
    -not (Test-Path -LiteralPath $buildTools) -or
    -not (Test-Path -LiteralPath $platform)) {
    throw "Android SDK is incomplete: $sdkRoot. Install cmdline-tools, platform-tools, platforms;android-36, and build-tools;36.0.0 first."
}

New-Item -ItemType Directory -Force -Path $gradleHome | Out-Null
$env:JAVA_HOME = $javaHome
$env:ANDROID_HOME = $sdkRoot
$env:ANDROID_SDK_ROOT = $sdkRoot
$env:GRADLE_USER_HOME = $gradleHome

$localProperties = Join-Path $PSScriptRoot "local.properties"
$sdkEscaped = $sdkRoot.Replace("\", "\\").Replace(":", "\:")
Set-Content -LiteralPath $localProperties -Value "sdk.dir=$sdkEscaped" -Encoding ASCII

$wrapper = Join-Path $PSScriptRoot "gradlew.bat"
Push-Location $PSScriptRoot
try {
    if (Test-Path -LiteralPath $wrapper) {
        & $wrapper :app:assembleDebug --no-daemon
    } else {
        $gradle = Get-Command gradle -ErrorAction SilentlyContinue
        if (-not $gradle) {
            throw "gradlew.bat or gradle was not found. Run the toolchain bootstrap script first."
        }
        & $gradle.Source :app:assembleDebug --no-daemon
    }
    $buildExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($buildExitCode -ne 0) {
    throw "Gradle build failed with exit code $buildExitCode."
}

$apk = Join-Path $PSScriptRoot "app\build\outputs\apk\debug\app-debug.apk"
if (-not (Test-Path -LiteralPath $apk)) {
    throw "Build completed but APK was not found: $apk"
}

if (-not $SkipCopy) {
    New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
    $destination = Join-Path $artifactRoot "pawprint-id-debug.apk"
    Copy-Item -LiteralPath $apk -Destination $destination -Force
    Write-Output "APK=$destination"
    $hash = Get-Sha256Hex $destination
    Write-Output "SHA256=$hash"
    $checksumFile = "$destination.sha256"
    Set-Content -LiteralPath $checksumFile -Value "$hash  pawprint-id-debug.apk" -Encoding ASCII
    Write-Output "CHECKSUM=$checksumFile"
} else {
    Write-Output "APK=$apk"
}
