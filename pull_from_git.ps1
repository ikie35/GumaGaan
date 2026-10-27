$PSScriptRootPath = $PSScriptRoot
if ($PSScriptRootPath) { Set-Location -Path $PSScriptRootPath }

Write-Host "[Git] Pulling latest changes from GitHub (origin main)..." -ForegroundColor Cyan
git pull origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "[Git] Pull completed successfully! 🎉" -ForegroundColor Green
    if (Test-Path "sync_raspi_deploy.ps1") {
        Write-Host "[Deploy] Refreshing deployment package..." -ForegroundColor Cyan
        powershell -ExecutionPolicy Bypass -File .\sync_raspi_deploy.ps1
    }
} else {
    Write-Host "[Git] Pull encountered an issue. Please review output above." -ForegroundColor Red
}
