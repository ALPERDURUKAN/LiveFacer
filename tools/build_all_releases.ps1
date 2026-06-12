param(
    [string]$ProjectRoot = "a:\LF",
    [string]$BaseOutputDir = "a:\LF\dist_releases"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$pythonExe = Join-Path $ProjectRoot "system\python\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Portable python not found: $pythonExe"
}

$packageScript = Join-Path $ProjectRoot "tools\package_protected.py"
if (-not (Test-Path $packageScript)) {
    throw "Packager script not found: $packageScript"
}

# --- Create Launchers Helper Function ---
function Create-Launchers {
    param([string]$TargetDir)
    
    $LauncherContentBase = "@echo off`nsetlocal EnableExtensions EnableDelayedExpansion`nset `"BASE_DIR=%~dp0`"`nif `"%BASE_DIR:~-1%`"==`"\`" set `"BASE_DIR=%BASE_DIR:~0,-1%`"`nset `"SYSTEM_DIR=%BASE_DIR%\system`"`nset `"appdata=%SYSTEM_DIR%\tmp`"`nset `"userprofile=%SYSTEM_DIR%\tmp`"`nset `"TEMP=%SYSTEM_DIR%\tmp`"`nset `"TMP=%SYSTEM_DIR%\tmp`"`nset `"PATH=%SYSTEM_DIR%\python;%SYSTEM_DIR%\ffmpeg;%SYSTEM_DIR%\CUDA;%SYSTEM_DIR%\CUDA\lib;%SYSTEM_DIR%\CUDA\bin;%PATH%`"`nset `"TF_CPP_MIN_LOG_LEVEL=2`"`n"
    
    # CPU Launcher
    $CpuContent = $LauncherContentBase + "`necho Starting on CPU...`n.\LiveFacerProtected.exe --execution-provider cpu %*`npause`n"
    Set-Content -Path (Join-Path $TargetDir "Run_CPU.bat") -Value $CpuContent
    
    # NVIDIA Launcher
    $NvidiaContent = $LauncherContentBase + "`nset `"CUDA_MODULE_LOADING=LAZY`"`necho Starting on NVIDIA GPU...`n.\LiveFacerProtected.exe --execution-provider cuda --execution-threads 1 %*`npause`n"
    Set-Content -Path (Join-Path $TargetDir "Run_NVIDIA.bat") -Value $NvidiaContent

    # AMD Launcher (DirectML)
    $AmdContent = $LauncherContentBase + "`necho Starting on AMD GPU...`n.\LiveFacerProtected.exe --execution-provider directml %*`npause`n"
    Set-Content -Path (Join-Path $TargetDir "Run_AMD.bat") -Value $AmdContent
}

function Copy-System-Deps {
    param([string]$TargetDir)
    Write-Host "[INFO] Copying system directories to $TargetDir..."
    $SystemTarget = Join-Path $TargetDir "system"
    New-Item -ItemType Directory -Force -Path $SystemTarget | Out-Null
    foreach ($dir in @("CUDA", "python", "ffmpeg", "git")) {
        $srcDir = Join-Path $ProjectRoot "system\$dir"
        $dstDir = Join-Path $SystemTarget $dir
        if (Test-Path $srcDir) {
            Write-Host "Copying $dir..."
            # Use robocopy for faster directory copying, suppress normal output
            robocopy $srcDir $dstDir /E /MT:8 /NFL /NDL /NJH /NJS /nc /ns /np
        }
    }
}

# 1. Build Licensed Version
$LicensedOutDir = Join-Path $BaseOutputDir "LiveFacer_Licensed"
Write-Host "======================================"
Write-Host "Building LICENSED version..."
Write-Host "======================================"
& $pythonExe $packageScript --project-root $ProjectRoot --output-dir $LicensedOutDir --minimal-stage
if ($LASTEXITCODE -ne 0) { throw "Licensed build failed" }
Create-Launchers -TargetDir (Join-Path $LicensedOutDir "bin\LiveFacerProtected")
Copy-System-Deps -TargetDir (Join-Path $LicensedOutDir "bin\LiveFacerProtected")

# 2. Build Unlicensed Version (Disabled to save disk space)
# $UnlicensedOutDir = Join-Path $BaseOutputDir "LiveFacer_Unlicensed"
# Write-Host "======================================"
# Write-Host "Building UNLICENSED version..."
# Write-Host "======================================"
# & $pythonExe $packageScript --project-root $ProjectRoot --output-dir $UnlicensedOutDir --no-license --minimal-stage
# if ($LASTEXITCODE -ne 0) { throw "Unlicensed build failed" }
# Create-Launchers -TargetDir (Join-Path $UnlicensedOutDir "bin\LiveFacerProtected")
# Copy-System-Deps -TargetDir (Join-Path $UnlicensedOutDir "bin\LiveFacerProtected")

Write-Host "======================================"
Write-Host "All builds completed successfully! Check the dist_releases folder."
Write-Host "======================================"
