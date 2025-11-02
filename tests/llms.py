"""
Test-local shim so `from llms import ...` works when tests modify sys.path.
Forwards to the real implementation in `src/core/llms.py`.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.llms import *  # noqa: F401,F403

