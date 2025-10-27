# Testing Guide

This guide explains how to run unit, integration, and full test suites for the project, plus tips for common issues on Windows.

Prerequisites
- Use the backend virtual environment (`venv`) with the backend/core dependencies installed (see README).
- Python 3.11 (project pins align with this).

Quick start (unit tests)
1) Activate backend env
```powershell
.\\venv\\Scripts\\Activate.ps1
```
2) Install minimal testing dependencies
```powershell
python -m pip install -U pytest pytest-asyncio pytest-mock pytest-xdist pytest-cov freezegun responses factory-boy
```
3) Run unit tests
```powershell
pytest -q tests\\unit
```

Full suite (tests + coverage + lint + security)
Option A: Use the test runner
```powershell
.\\venv\\Scripts\\Activate.ps1
# First time only (installs test toolchain like black, isort, flake8, bandit, safety)
python run_tests.py --install
# Run everything: tests, linting, security; generates reports under reports/ and htmlcov/
python run_tests.py --all
```

Option B: Manual commands
```powershell
.\\venv\\Scripts\\Activate.ps1
# Install tools
python -m pip install -U pytest pytest-asyncio pytest-mock pytest-xdist pytest-cov pytest-html pytest-json-report black isort flake8 bandit safety

# Run test suite with coverage and reports
pytest tests \\
  --cov=src --cov-report=html:htmlcov --cov-report=term-missing \\
  --html=reports/pytest_report.html --self-contained-html \\
  --json-report --json-report-file=reports/pytest_report.json \\
  --maxfail=5 --durations=10
```

Environment variables for integration/e2e
- `GROQ_API_KEY` must be set for LLM-backed tests. Either set it in your shell or put it in `.env`.
- The test runner sets `LANGCHAIN_TRACING_V2=false` by default.

Common issues and fixes
- Windows console encoding errors when printing emoji
  - Run with UTF-8 mode: `setx PYTHONUTF8 1` (new terminals) or `python -X utf8 ...`
- Missing coverage.exe or permission errors
  - Ensure you are using the project venv’s Python: `where.exe python`
  - Use `python -m pip install ...` to target the venv explicitly
- Unknown pytest marks (unit, math, spanish, etc.)
  - The suite registers marks in `pytest.ini`; if you still see warnings, ensure Pytest loads `pytest.ini` from the repo root
- Import errors in legacy tests (e.g., `from llms import ...`)
  - Use the backend venv and run from the repo root so `src/` is on the path via pytest config
  - If a specific legacy test still imports `llms` from top-level, run unit tests only or skip that test: `pytest -k "not test_multi_llm"`

Reports
- Coverage HTML: `htmlcov/index.html`
- Pytest HTML: `reports/pytest_report.html`
- Pytest JSON: `reports/pytest_report.json`
- Security: `reports/bandit_report.json`, `reports/safety_report.json` (when using the runner)

Tips
- Start with unit tests; then run the full suite once dependencies and keys are set
- Use `-n auto` to parallelize on larger machines: `pytest -n auto tests`

