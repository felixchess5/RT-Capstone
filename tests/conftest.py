"""
Pytest configuration and shared fixtures for Intelligent-Assignment-Grading-System tests.

This module provides common fixtures, test utilities, and configuration
for all test modules in the test suite.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
import pytest
from freezegun import freeze_time

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.assignment_orchestrator import AssignmentOrchestrator

# Import project modules
from core.llms import MultiLLMManager
from processors.history_processor import HistoryProcessor
from processors.math_processor import MathProcessor
from processors.science_processor import ScienceProcessor
from processors.spanish_processor import SpanishProcessor
from support.file_processor import FileProcessor
from support.language_support import LanguageManager
from workflows.agentic_workflow import build_agentic_workflow, create_workflow

# ========== PYTEST CONFIGURATION ==========


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["PYTEST_CURRENT_TEST"] = "true"


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and paths."""
    for item in items:
        # Add markers based on test file names
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in item.nodeid or "end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.e2e)

        # Add component-specific markers
        if "math" in item.nodeid.lower():
            item.add_marker(pytest.mark.math)
        elif "spanish" in item.nodeid.lower():
            item.add_marker(pytest.mark.spanish)
        elif "science" in item.nodeid.lower():
            item.add_marker(pytest.mark.science)
        elif "history" in item.nodeid.lower():
            item.add_marker(pytest.mark.history)
        elif "gradio" in item.nodeid.lower():
            item.add_marker(pytest.mark.gradio)
        elif "mcp" in item.nodeid.lower():
            item.add_marker(pytest.mark.mcp)

        # Add asyncio marker for async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


# ========== ENVIRONMENT FIXTURES ==========


@pytest.fixture(scope="session")
def test_env():
    """Set up test environment variables."""
    env_vars = {
        "GROQ_API_KEY": "test_groq_key",
        "GEMINI_API_KEY": "test_gemini_key",
        "LANGCHAIN_TRACING_V2": "false",
        "TESTING": "true",
    }

    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield env_vars

    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_files(temp_dir):
    """Create temporary test files."""
    files = {}

    # Sample assignment content
    math_content = """
    Name: John Doe
    Date: 2025-01-15
    Class: Algebra II
    Subject: Mathematics

    Problem 1: Solve for x: 2x + 5 = 13
    Solution: x = 4

    Problem 2: Factor: x² - 5x + 6
    Solution: (x-2)(x-3)
    """

    spanish_content = """
    Name: Maria Garcia
    Date: 2025-01-15
    Class: Spanish III
    Subject: Spanish

    Ensayo sobre la cultura hispana:
    La cultura hispana es muy rica y diversa. Incluye tradiciones
    de muchos países diferentes como España, México, Argentina...
    """

    science_content = """
    Name: Alex Johnson
    Date: 2025-01-15
    Class: Biology
    Subject: Science

    Lab Report: Photosynthesis Experiment
    Hypothesis: Plants produce oxygen during photosynthesis.
    Procedure: 1. Place aquatic plant in water...
    Results: Oxygen bubbles observed...
    Conclusion: Hypothesis confirmed.
    """

    # Create test files
    files["math"] = temp_dir / "math_assignment.txt"
    files["math"].write_text(math_content)

    files["spanish"] = temp_dir / "spanish_assignment.txt"
    files["spanish"].write_text(spanish_content)

    files["science"] = temp_dir / "science_assignment.txt"
    files["science"].write_text(science_content)

    files["empty"] = temp_dir / "empty.txt"
    files["empty"].write_text("")

    files["invalid"] = temp_dir / "invalid.xyz"
    files["invalid"].write_text("Invalid file type")

    yield files


# ========== MOCK FIXTURES ==========


@pytest.fixture
def mock_llm():
    """Mock LLM for testing without API calls."""
    mock = AsyncMock()
    mock.ainvoke.return_value = Mock(content="Mock LLM response")
    mock.invoke.return_value = Mock(content="Mock LLM response")
    return mock


@pytest.fixture
def mock_multi_llm_manager(mock_llm):
    """Mock MultiLLMManager for testing."""
    with patch("core.llms.MultiLLMManager") as mock_class:
        mock_instance = Mock()
        mock_instance.get_llm.return_value = mock_llm
        mock_instance.get_available_llms.return_value = ["groq", "gemini"]
        mock_instance.health_check.return_value = {"groq": True, "gemini": True}
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_file_processor():
    """Mock FileProcessor for testing."""
    mock = Mock()
    mock.extract_text_content.return_value = "Sample assignment text"
    mock.is_valid_file.return_value = True
    mock.get_file_type.return_value = "txt"
    return mock


@pytest.fixture
def mock_language_support():
    """Mock LanguageManager for testing."""
    mock = Mock()
    mock.detect_language.return_value = "en"
    mock.get_localized_prompt.return_value = "Localized prompt"
    mock.get_supported_languages.return_value = ["en", "es", "fr"]
    return mock


# ========== COMPONENT FIXTURES ==========


@pytest.fixture
def math_processor(mock_multi_llm_manager):
    """Create MathProcessor instance for testing."""
    return MathProcessor(mock_multi_llm_manager)


@pytest.fixture
def spanish_processor(mock_multi_llm_manager):
    """Create SpanishProcessor instance for testing."""
    return SpanishProcessor(mock_multi_llm_manager)


@pytest.fixture
def science_processor(mock_multi_llm_manager):
    """Create ScienceProcessor instance for testing."""
    return ScienceProcessor(mock_multi_llm_manager)


@pytest.fixture
def history_processor(mock_multi_llm_manager):
    """Create HistoryProcessor instance for testing."""
    return HistoryProcessor(mock_multi_llm_manager)


@pytest.fixture
def assignment_orchestrator(mock_multi_llm_manager):
    """Create AssignmentOrchestrator instance for testing."""
    return AssignmentOrchestrator(mock_multi_llm_manager)


@pytest.fixture
def file_processor():
    """Create FileProcessor instance for testing."""
    return FileProcessor()


# ========== DATA FIXTURES ==========


@pytest.fixture
def sample_math_assignment():
    """Sample math assignment data."""
    return {
        "text": "Solve for x: 2x + 5 = 13\nSolution: x = 4",
        "metadata": {
            "name": "John Doe",
            "date": "2025-01-15",
            "class": "Algebra II",
            "subject": "Mathematics",
        },
    }


@pytest.fixture
def sample_spanish_assignment():
    """Sample Spanish assignment data."""
    return {
        "text": "Ensayo sobre la cultura hispana: La cultura es muy rica...",
        "metadata": {
            "name": "Maria Garcia",
            "date": "2025-01-15",
            "class": "Spanish III",
            "subject": "Spanish",
        },
    }


@pytest.fixture
def sample_results():
    """Sample grading results."""
    return {
        "overall_score": 8.5,
        "grammar": {"score": 8, "errors": 2},
        "plagiarism": {"score": 9, "percentage": 5},
        "relevance": {"score": 8, "alignment": 80},
        "summary": "Good assignment with minor grammar issues",
    }


@pytest.fixture
def sample_batch_data():
    """Sample batch processing data."""
    return [
        {
            "file_path": "assignment1.txt",
            "content": "Math assignment content",
            "subject": "Mathematics",
        },
        {
            "file_path": "assignment2.txt",
            "content": "Spanish assignment content",
            "subject": "Spanish",
        },
        {
            "file_path": "assignment3.txt",
            "content": "Science assignment content",
            "subject": "Science",
        },
    ]


# ========== ASYNC FIXTURES ==========


@pytest.fixture
async def async_workflow(mock_multi_llm_manager):
    """Create async workflow for testing."""
    workflow = create_workflow(mock_multi_llm_manager)
    yield workflow


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ========== TIME FIXTURES ==========


@pytest.fixture
def fixed_time():
    """Freeze time for consistent testing."""
    with freeze_time("2025-01-15 12:00:00"):
        yield


# ========== GRADIO FIXTURES ==========


@pytest.fixture
def mock_gradio_interface():
    """Mock Gradio interface for testing."""
    mock = Mock()
    mock.launch.return_value = None
    mock.close.return_value = None
    return mock


# ========== MCP FIXTURES ==========


@pytest.fixture
def mock_mcp_server():
    """Mock MCP server for testing."""
    mock = Mock()
    mock.start.return_value = None
    mock.stop.return_value = None
    mock.handle_request.return_value = {"result": "success"}
    return mock


# ========== UTILITY FUNCTIONS ==========


def create_test_assignment(subject: str = "Mathematics", **kwargs) -> Dict[str, Any]:
    """Create a test assignment with default values."""
    defaults = {
        "text": f"Sample {subject} assignment content",
        "metadata": {
            "name": "Test Student",
            "date": "2025-01-15",
            "class": f"{subject} Class",
            "subject": subject,
        },
    }
    defaults.update(kwargs)
    return defaults


def assert_grading_result(
    result: Dict[str, Any], min_score: float = 0.0, max_score: float = 10.0
):
    """Assert that grading result has expected structure and values."""
    assert isinstance(result, dict)
    assert "overall_score" in result
    assert min_score <= result["overall_score"] <= max_score

    # Check required fields
    required_fields = ["grammar", "plagiarism", "relevance", "summary"]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"


def assert_classification_result(result: Dict[str, Any]):
    """Assert that classification result has expected structure."""
    assert isinstance(result, dict)
    assert "subject" in result
    assert "complexity" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0


# ========== PERFORMANCE FIXTURES ==========


@pytest.fixture
def performance_monitor():
    """Monitor test performance."""
    import time

    start_time = time.time()
    yield
    end_time = time.time()
    duration = end_time - start_time
    if duration > 5.0:  # Warn if test takes more than 5 seconds
        pytest.warn(f"Test took {duration:.2f} seconds (> 5s threshold)")


# ========== CLEANUP ==========


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    yield
    # Cleanup any test files created during testing
    test_patterns = ["test_*.tmp", "*.test", "test_output_*"]
    for pattern in test_patterns:
        for file_path in Path(".").glob(pattern):
            try:
                file_path.unlink()
            except (OSError, FileNotFoundError):
                pass
