"""
Public shim for the multi-provider LLM manager.

This module allows imports like `from llms import llm_manager, invoke_with_fallback`
without exposing the internal `src/core` layout.

To be friendlier in production contexts we:
- Resolve the `src` directory using absolute paths.
- Avoid duplicate `sys.path` entries.
- Surface a clear ImportError if the underlying core module cannot be loaded.
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
    logger.error("Expected src directory to exist at %s", SRC_DIR)

try:
    from core.llms import *  # type: ignore  # noqa: F401,F403
except Exception as exc:  # pragma: no cover - defensive import wrapper
    logger.exception("Failed to import core.llms via shim")
    raise ImportError(
        "Could not import 'core.llms'. "
        "Ensure you are running from the project root and that "
        "'src/core/llms.py' exists and its dependencies are installed."
    ) from exc
