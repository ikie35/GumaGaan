param(
    [string]$Message = "chore: update rubber disease system"
)

$PSScriptRootPath = $PSScriptRoot
if ($PSScriptRootPath) { Set-Location -Path $PSScriptRootPath }

# Ensure deployment package is synced before commit
if (Test-Path "sync_raspi_deploy.ps1") {
    powershell -ExecutionPolicy Bypass -File .\sync_raspi_deploy.ps1
}

Write-Host "[Git] Staging files..." -ForegroundColor Cyan
git add -A

$status = git status --porcelain
if (-not $status) {
    Write-Host "[Git] No changes to commit. Working tree is clean." -ForegroundColor Green
    exit 0
}

Write-Host "[Git] Committing changes with message: '$Message'..." -ForegroundColor Cyan
git commit -m "$Message"

Write-Host "[Git] Pushing to GitHub (origin main)..." -ForegroundColor Cyan
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "[Git] Push completed successfully! 🎉" -ForegroundColor Green
} else {
    Write-Host "[Git] Push encountered an issue. Please review output above." -ForegroundColor Red
}
