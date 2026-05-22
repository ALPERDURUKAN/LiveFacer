param(
    [string]$ProjectRoot = "a:\LF",
    [string]$OutputDir = "",
    [switch]$SkipPyInstaller = $false,
    [switch]$MinimalStage = $false,
    [string]$ProductId = "",
    [string]$ProductPermalink = "",
    [int]$GraceDays = 7
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectRoot "dist_protected"
}

$pythonExe = Join-Path $ProjectRoot "system\python\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Portable python not found: $pythonExe"
}

$packageScript = Join-Path $ProjectRoot "tools\package_protected.py"
if (-not (Test-Path $packageScript)) {
    throw "Packager script not found: $packageScript"
}

$scriptArgs = @(
    $packageScript,
    "--project-root", $ProjectRoot,
    "--output-dir", $OutputDir,
    "--grace-days", $GraceDays
)

if ($SkipPyInstaller) {
    $scriptArgs += "--skip-pyinstaller"
}
if ($MinimalStage) {
    $scriptArgs += "--minimal-stage"
}
if ($ProductId) {
    $scriptArgs += @("--product-id", $ProductId)
}
if ($ProductPermalink) {
    $scriptArgs += @("--product-permalink", $ProductPermalink)
}

Write-Host "[INFO] Starting protected packaging pipeline..."
Write-Host "[INFO] Project root: $ProjectRoot"
Write-Host "[INFO] Output dir  : $OutputDir"
if ($SkipPyInstaller) {
    Write-Host "[INFO] PyInstaller step will be skipped."
}

& $pythonExe @scriptArgs
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    throw "Protected packaging failed with exit code $exitCode"
}

Write-Host "[OK] Protected packaging completed."
