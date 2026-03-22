# PowerShell script to start ICU Prediction System

$ErrorActionPreference = "Continue"

Write-Host "======================================================================"
Write-Host "ICU MULTI-MODAL PREDICTION SYSTEM - Starting..."
Write-Host "======================================================================"
Write-Host ""

# Change to project directory
Set-Location E:\icu_project

Write-Host "[1] Checking Python...
"
& "E:\ANACONDA\envs\icu_project\python.exe" --version

Write-Host "[2] Starting Flask server..." -ForegroundColor Green
Write-Host ""
Write-Host "System will be available at: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""

# Run the app
& "E:\ANACONDA\envs\icu_project\python.exe" app.py
