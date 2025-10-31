"""
Shim module to allow `from assignment_orchestrator import ...` imports.

This adjusts `sys.path` so modules under `src/core/` can be imported
without requiring callers to set PYTHONPATH. It is used by examples and tests.
"""

import os
import sys

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.assignment_orchestrator import *  # noqa: F401,F403

