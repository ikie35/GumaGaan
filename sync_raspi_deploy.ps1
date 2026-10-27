# Synchronize latest stable files, ONNX, and Hailo HEF models into standalone raspi_deploy folder

$PSScriptRootPath = $PSScriptRoot
if ($PSScriptRootPath) { Set-Location -Path $PSScriptRootPath }

$DeployDir = Join-Path $PSScriptRootPath "raspi_deploy"
$ModelsDir = Join-Path $DeployDir "models"

Write-Host "Creating deployment folders at: $DeployDir" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $DeployDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

Write-Host "Copying core application files..." -ForegroundColor Cyan
Copy-Item -Path "app.py" -Destination $DeployDir -Force
Copy-Item -Path "raspi_capture.py" -Destination $DeployDir -Force
Copy-Item -Path "collect_dataset.py" -Destination $DeployDir -Force
Copy-Item -Path "static" -Destination $DeployDir -Recurse -Force
Copy-Item -Path "templates" -Destination $DeployDir -Recurse -Force

Write-Host "Copying Stage 1 Detector Hailo HEF model (.hef)..." -ForegroundColor Cyan
# Stage 1 YOLOv8n single-class leaf detector HEF
$S1Hef = "Stage 1 Rubber Leaf Detector/HEF models/stage1_single_class_yolov8n_best_hailo_model/stage1_single_class_yolov8n_best.hef"
if (Test-Path $S1Hef) { 
    Copy-Item -Path $S1Hef -Destination "$ModelsDir/stage1_detector.hef" -Force
    Write-Host "  -> stage1_detector.hef copied ($S1Hef)" -ForegroundColor Green 
} else {
    Write-Host "  -> WARNING: $S1Hef not found!" -ForegroundColor Red
}

Write-Host "Copying Stage 2 Classifier Hailo HEF model (.hef)..." -ForegroundColor Cyan
# Stage 2 YOLO11n-cls Field-Robust HEF (yolo11n_robust_best.hef)
$S2Hef = "Stage2_Classifier/yolo11n-clsHEF/yolo11n-cls_best_hailo_model/yolo11n_robust_best.hef"
if (-not (Test-Path $S2Hef)) { $S2Hef = "Stage2_Classifier/yolo11n-clsHEF/yolo11n-cls_best_hailo_model/yolo11n-cls_best.hef" }
if (-not (Test-Path $S2Hef)) { $S2Hef = "Stage2_Classifier/stage2best/stage2_best_hailo_model/stage2_best.hef" }
if (Test-Path $S2Hef) { 
    Copy-Item -Path $S2Hef -Destination "$ModelsDir/stage2_classifier.hef" -Force
    Write-Host "  -> stage2_classifier.hef copied ($S2Hef)" -ForegroundColor Green 
} else {
    Write-Host "  -> WARNING: $S2Hef not found!" -ForegroundColor Red
}

# Ensure no stale ONNX files linger in deployment
Get-ChildItem -Path $ModelsDir -Filter "*.onnx" -ErrorAction SilentlyContinue | Remove-Item -Force

Write-Host "Synchronization complete! Pure Hailo-8L .hef models prepared in 'raspi_deploy/models'." -ForegroundColor Green
