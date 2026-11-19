# One-time setup for Windows 10/11 (PowerShell). Run from the repo root:  .\scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\backend"
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
if (-not (Test-Path ".env")) {
    Copy-Item ..\.env.example .env
    # Offline SQLite so nothing else needs to be installed for the first run
    (Get-Content .env) -replace '^DATABASE_URL=.*', 'DATABASE_URL=sqlite:///./data/researchpilot.db' | Set-Content .env
}
& .\.venv\Scripts\python.exe -m alembic upgrade head
Write-Host "Setup complete. Next: .\scripts\check.ps1"
