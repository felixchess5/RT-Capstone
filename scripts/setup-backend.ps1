param(
  [string]$VenvPath = "venv"
)

$ErrorActionPreference = 'Stop'

Write-Host "[backend] Creating venv at $VenvPath"
python -m venv $VenvPath

Write-Host "[backend] Activating venv and installing core requirements"
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvPath\Scripts\python.exe" -m pip install -r requirements-core.txt

Write-Host "[backend] Verifying key packages"
@'
import importlib.metadata as m
def v(name):
    try:
        print(name, m.version(name))
    except Exception:
        print(name, 'not installed')
for pkg in ['fastapi','uvicorn','langchain-core','langchain-openai','langchain-groq','spacy','typer']:
    v(pkg)
'@ | & "$VenvPath\Scripts\python.exe" -

Write-Host "[backend] Done. Activate with: `n  .\\$VenvPath\\Scripts\\Activate.ps1"
