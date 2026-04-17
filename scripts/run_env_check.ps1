param(
  [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $VenvPath)) {
  Write-Host "Venv not found at $VenvPath. Create it first." -ForegroundColor Yellow
  exit 1
}

& "$VenvPath\Scripts\python.exe" -m src.cli_env_check
