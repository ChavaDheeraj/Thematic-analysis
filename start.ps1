# Start the ThematicML web application on http://127.0.0.1:8000
# Usage: .\start.ps1 [-Port 8000]
param([int]$Port = 8000)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
if (-not (Test-Path (Join-Path $root "frontend\dist\index.html"))) {
    Write-Host "Building the web UI (first run)..."
    Push-Location (Join-Path $root "frontend"); npm install; npm run build; Pop-Location
}
Write-Host "ThematicML running at http://127.0.0.1:$Port  (Ctrl+C to stop)"
# --no-access-log: request URLs can contain search terms and record filters typed by the researcher
& $py -m uvicorn thematicml_api.main:app --host 127.0.0.1 --port $Port --no-access-log
