"""
Centralized path configuration for the grading system.

These constants are used across the project to locate input assignments,
outputs, and generated artifacts. They are defined using absolute paths
rooted at the project directory to avoid surprises when the current
working directory changes (e.g., under different process managers).

Values can be overridden with environment variables where appropriate
to support different deployment layouts without code changes.
"""

from __future__ import annotations

import os
from typing import List


def _project_root() -> str:
    """Resolve the absolute project root directory.

    This is computed relative to this file to remain stable even if
    the process working directory changes.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    # src/core/ -> project root
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


PROJECT_ROOT: str = _project_root()

# Base folders
ASSIGNMENTS_FOLDER: str = os.getenv(
    "ASSIGNMENTS_FOLDER", os.path.join(PROJECT_ROOT, "Assignments")
)
OUTPUT_FOLDER: str = os.getenv(
    "OUTPUT_FOLDER", os.path.join(PROJECT_ROOT, "output")
)
PLAGIARISM_REPORTS_FOLDER: str = os.getenv(
    "PLAGIARISM_REPORTS_FOLDER", os.path.join(PROJECT_ROOT, "plagiarism_reports")
)

# Top-level artifacts
GRAPH_OUTPUT_PATH: str = os.getenv(
    "GRAPH_OUTPUT_PATH", os.path.join(PROJECT_ROOT, "graph.png")
)
SUMMARY_CSV_PATH: str = os.path.join(OUTPUT_FOLDER, "summary.csv")

# Subject-specific output paths (CSV)
MATH_CSV_PATH: str = os.path.join(OUTPUT_FOLDER, "math_assignments.csv")
SPANISH_CSV_PATH: str = os.path.join(OUTPUT_FOLDER, "spanish_assignments.csv")
ENGLISH_CSV_PATH: str = os.path.join(OUTPUT_FOLDER, "english_assignments.csv")
SCIENCE_CSV_PATH: str = os.path.join(OUTPUT_FOLDER, "science_assignments.csv")
HISTORY_CSV_PATH: str = os.path.join(OUTPUT_FOLDER, "history_assignments.csv")
GENERAL_CSV_PATH: str = os.path.join(OUTPUT_FOLDER, "general_assignments.csv")

# Subject-specific JSON paths
MATH_JSON_PATH: str = os.path.join(OUTPUT_FOLDER, "math_assignments.json")
SPANISH_JSON_PATH: str = os.path.join(OUTPUT_FOLDER, "spanish_assignments.json")
ENGLISH_JSON_PATH: str = os.path.join(OUTPUT_FOLDER, "english_assignments.json")
SCIENCE_JSON_PATH: str = os.path.join(OUTPUT_FOLDER, "science_assignments.json")
HISTORY_JSON_PATH: str = os.path.join(OUTPUT_FOLDER, "history_assignments.json")
GENERAL_JSON_PATH: str = os.path.join(OUTPUT_FOLDER, "general_assignments.json")

REQUIRED_FOLDERS: List[str] = [
    ASSIGNMENTS_FOLDER,
    OUTPUT_FOLDER,
    PLAGIARISM_REPORTS_FOLDER,
]
