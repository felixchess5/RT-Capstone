param(
  [string]$VenvPath = ".venv-demo"
)

$ErrorActionPreference = 'Stop'

Write-Host "[demo] Creating venv at $VenvPath"
python -m venv $VenvPath

Write-Host "[demo] Activating venv and installing demo requirements"
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvPath\Scripts\python.exe" -m pip install -r requirements-demo.txt

Write-Host "[demo] Verifying key packages"
@'
import importlib.metadata as m
def v(name):
    try:
        print(name, m.version(name))
    except Exception:
        print(name, 'not installed')
for pkg in ['gradio','gradio_client','httpx','typer','pandas']:
    v(pkg)
'@ | & "$VenvPath\Scripts\python.exe" -

Write-Host "[demo] Done. Activate with: `n  .\\$VenvPath\\Scripts\\Activate.ps1"
