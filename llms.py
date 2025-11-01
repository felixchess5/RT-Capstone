"""
Shim module to allow `from llms import ...` imports used in tests/examples.
Forwards to the real implementation in `src/core/llms.py`.
"""

import os
import sys

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.llms import *  # noqa: F401,F403

