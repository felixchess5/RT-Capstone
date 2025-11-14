import os
from pathlib import Path

from core import paths


def test_project_root_points_to_repo_root():
    current = Path(__file__).resolve()
    # tests/unit -> tests -> repo root
    expected_root = current.parents[2]
    assert Path(paths.PROJECT_ROOT).resolve() == expected_root


def test_default_folders_are_under_project_root(monkeypatch):
    # Ensure env vars do not interfere
    monkeypatch.delenv("ASSIGNMENTS_FOLDER", raising=False)
    monkeypatch.delenv("OUTPUT_FOLDER", raising=False)
    monkeypatch.delenv("PLAGIARISM_REPORTS_FOLDER", raising=False)

    # Reload module to pick up cleared env vars
    import importlib

    import core.paths as paths_module

    importlib.reload(paths_module)

    root = Path(paths_module.PROJECT_ROOT)
    assert Path(paths_module.ASSIGNMENTS_FOLDER).is_relative_to(root)
    assert Path(paths_module.OUTPUT_FOLDER).is_relative_to(root)
    assert Path(paths_module.PLAGIARISM_REPORTS_FOLDER).is_relative_to(root)


def test_env_overrides_take_precedence(monkeypatch, tmp_path):
    assignments_dir = tmp_path / "assignments"
    output_dir = tmp_path / "out"
    reports_dir = tmp_path / "reports"

    monkeypatch.setenv("ASSIGNMENTS_FOLDER", str(assignments_dir))
    monkeypatch.setenv("OUTPUT_FOLDER", str(output_dir))
    monkeypatch.setenv("PLAGIARISM_REPORTS_FOLDER", str(reports_dir))

    import importlib

    import core.paths as paths_module

    importlib.reload(paths_module)

    assert Path(paths_module.ASSIGNMENTS_FOLDER) == assignments_dir
    assert Path(paths_module.OUTPUT_FOLDER) == output_dir
    assert Path(paths_module.PLAGIARISM_REPORTS_FOLDER) == reports_dir
    # Summary CSV should be under the overridden output directory
    assert Path(paths_module.SUMMARY_CSV_PATH).parent == output_dir

