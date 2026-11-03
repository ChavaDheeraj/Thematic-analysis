# Start the ThematicML web application.
# Usage: .\start.ps1 [-Port 8000]       this computer only: http://127.0.0.1:8000
#        .\start.ps1 -Network           also phones and laptops on the same network, behind a password
param([int]$Port = 8000, [switch]$Network)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
if (-not (Test-Path (Join-Path $root "frontend\dist\index.html"))) {
    Write-Host "Building the web UI (first run)..."
    Push-Location (Join-Path $root "frontend"); npm install; npm run build; Pop-Location
}
$bind = "127.0.0.1"
if ($Network) {
    if (-not $env:THEMATICML_PASSWORD) {
        $secure = Read-Host "Choose a password for other devices" -AsSecureString
        $env:THEMATICML_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
    }
    if (-not $env:THEMATICML_PASSWORD) { throw "Other devices can only connect once a password is set." }
    $env:THEMATICML_REQUIRE_PASSWORD = "1"
    $bind = "0.0.0.0"
    $user = if ($env:THEMATICML_USER) { $env:THEMATICML_USER } else { "researcher" }
    $ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } | Select-Object -ExpandProperty IPAddress
    Write-Host "Other devices on this network can open:"
    foreach ($ip in $ips) { Write-Host "    http://${ip}:$Port" }
    Write-Host "Sign in as '$user' with the password you chose. If Windows asks, allow Python on private networks."
}
Write-Host "ThematicML running at http://127.0.0.1:$Port  (Ctrl+C to stop)"
# --no-access-log: request URLs can contain search terms and record filters typed by the researcher
& $py -m uvicorn thematicml_api.main:app --host $bind --port $Port --no-access-log
