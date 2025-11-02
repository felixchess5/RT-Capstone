"""
Shim module to allow tests importing `history_processor` from the repository root.
This forwards all public names from the real implementation in src/processors.
"""

from src.processors.history_processor import *  # noqa: F401,F403

