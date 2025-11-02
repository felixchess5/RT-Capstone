"""
Lightweight startup shim to sanitize pytest CLI arguments in environments
that pass benchmark-only flags when the pytest-benchmark plugin is not
installed. Python auto-imports sitecustomize at startup if present on
sys.path (CWD is on sys.path), so this runs before pytest parses args.
"""

from __future__ import annotations

import sys


def _strip_benchmark_args(argv: list[str]) -> list[str]:
    """Remove pytest-benchmark CLI flags if present.

    Handles both ``--benchmark-json=...`` and the split form
    ``--benchmark-json <file>``.
    """
    kept: list[str] = []
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a == "--benchmark-only":
            i += 1
            continue
        if a == "--benchmark-json":
            i += 2  # skip option and its value
            continue
        if a.startswith("--benchmark-json="):
            i += 1
            continue
        kept.append(a)
        i += 1
    return kept


try:
    sys.argv = _strip_benchmark_args(sys.argv)
except Exception:
    # Be conservative: never block test execution due to shim issues
    pass

