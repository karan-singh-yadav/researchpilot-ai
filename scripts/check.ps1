# Lint + tests + offline demo. Run from the repo root:  .\scripts\check.ps1
$ErrorActionPreference = "Stop"
$py = "$PSScriptRoot\..\backend\.venv\Scripts\python.exe"
Push-Location "$PSScriptRoot\..\backend"
& $py -m ruff check --config pyproject.toml . ../scripts
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
Pop-Location
& $py "$PSScriptRoot\demo_pipeline.py" --depth standard
