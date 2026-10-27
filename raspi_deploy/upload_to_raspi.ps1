# Configuration for Raspberry Pi ONNX Deployment
$PiIP = "192.168.137.170"
$PiUser = "ikee"
$DestDir = "/home/ikee/Two stage Rubber Disease"

$PSScriptRootPath = $PSScriptRoot
if ($PSScriptRootPath) { Set-Location -Path $PSScriptRootPath }

Write-Host "Creating target directories on Raspberry Pi..." -ForegroundColor Cyan
ssh "${PiUser}@${PiIP}" "mkdir -p '$DestDir/static' '$DestDir/templates' '$DestDir/models'"

Write-Host "Uploading application core files..." -ForegroundColor Cyan
scp app.py "${PiUser}@${PiIP}:$DestDir/"
scp raspi_capture.py "${PiUser}@${PiIP}:$DestDir/"
scp -r static/* "${PiUser}@${PiIP}:$DestDir/static/"
scp -r templates/* "${PiUser}@${PiIP}:$DestDir/templates/"

Write-Host "Uploading ONLY Hailo HEF models (*.hef)..." -ForegroundColor Cyan
scp models/*.hef "${PiUser}@${PiIP}:$DestDir/models/"

Write-Host "Upload completed successfully! Pure Hailo-8L HEF models were deployed to Raspberry Pi." -ForegroundColor Green
