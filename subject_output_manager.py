"""
Public shim for subject output management helpers.

This module exists so callers can do `from subject_output_manager import ...`
without having to know about the internal `src/core` layout or adjust
`PYTHONPATH` manually.

In production-like environments we:
- Resolve the `src` directory robustly.
- Avoid mutating `sys.path` unnecessarily.
- Fail fast with a clear, actionable error message if the core module
  cannot be imported.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if SRC_DIR.is_dir():
    src_str = str(SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
else:
    # Log but still attempt the import – it will fail with a clearer error below
    logger.error("Expected src directory to exist at %s", SRC_DIR)

try:
    from core.subject_output_manager import *  # type: ignore  # noqa: F401,F403
except Exception as exc:  # pragma: no cover - defensive import wrapper
    logger.exception("Failed to import core.subject_output_manager via shim")
    raise ImportError(
        "Could not import 'core.subject_output_manager'. "
        "Ensure you are running from the project root and that "
        "'src/core/subject_output_manager.py' exists and is importable."
    ) from exc
