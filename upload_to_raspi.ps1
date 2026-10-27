param(
    [string]$TargetIP = ""
)

# Configuration for Raspberry Pi Hailo NPU & ONNX Deployment
$PiUser = "ikee"
$DestDir = "/home/ikee/Two stage Rubber Disease"

# Auto-detect Pi IP (Hotspot mode 10.42.0.1 or Local Network 192.168.137.183)
if ($TargetIP -ne "") {
    $PiIP = $TargetIP
    Write-Host "[Network] Using manually specified IP: $PiIP" -ForegroundColor Cyan
} elseif (Test-Connection -ComputerName "10.42.0.1" -Count 1 -Quiet) {
    $PiIP = "10.42.0.1"
    Write-Host "[Network] Detected Raspberry Pi in Hotspot Mode (10.42.0.1)!" -ForegroundColor Green
} elseif (Test-Connection -ComputerName "192.168.137.183" -Count 1 -Quiet) {
    $PiIP = "192.168.137.183"
    Write-Host "[Network] Detected Raspberry Pi at 192.168.137.183!" -ForegroundColor Cyan
} else {
    $PiIP = "10.42.0.1"
    Write-Host "[Network] Could not ping, defaulting to Hotspot IP: $PiIP" -ForegroundColor Yellow
}

$PSScriptRootPath = $PSScriptRoot
if ($PSScriptRootPath) { Set-Location -Path $PSScriptRootPath }

# Ensure raspi_deploy is up to date
if (Test-Path "sync_raspi_deploy.ps1") {
    powershell -ExecutionPolicy Bypass -File .\sync_raspi_deploy.ps1
}

$DeployDir = Join-Path $PSScriptRootPath "raspi_deploy"

Write-Host "Creating target directories on Raspberry Pi..." -ForegroundColor Cyan
ssh "${PiUser}@${PiIP}" "mkdir -p '$DestDir/static' '$DestDir/templates' '$DestDir/models'"

Write-Host "Uploading application core files & stress test suite..." -ForegroundColor Cyan
scp "$DeployDir/app.py" "${PiUser}@${PiIP}:$DestDir/"
scp "$DeployDir/raspi_capture.py" "${PiUser}@${PiIP}:$DestDir/"
scp "$DeployDir/collect_dataset.py" "${PiUser}@${PiIP}:$DestDir/"
scp "$DeployDir/stress_test.py" "${PiUser}@${PiIP}:$DestDir/"
scp -r "$DeployDir/static/*" "${PiUser}@${PiIP}:$DestDir/static/"
scp -r "$DeployDir/templates/*" "${PiUser}@${PiIP}:$DestDir/templates/"

Write-Host "Uploading pure Hailo HEF (.hef) models..." -ForegroundColor Cyan
scp "$DeployDir/models/*" "${PiUser}@${PiIP}:$DestDir/models/"

Write-Host "Upload completed successfully! Pure Hailo-8L HEF models deployed to Raspberry Pi." -ForegroundColor Green
