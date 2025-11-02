"""
Shim module so tests importing `agentic_workflow` at the repo root work.
Forwards to the real implementation in `src/workflows/agentic_workflow.py`.
"""

import os
import sys

ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from workflows.agentic_workflow import *  # noqa: F401,F403

