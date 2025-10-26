"""
Test script for MCP tools functionality.
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

# Test if MCP is available
try:
    from mcp_server import (
        grade_assignment,
        grammar_check,
        plagiarism_check,
        process_assignment_parallel,
        relevance_check,
        summarize_assignment,
    )

    MCP_AVAILABLE = True
    print("MCP tools imported successfully")
except ImportError as e:
    print(f"MCP import failed: {e}")
    MCP_AVAILABLE = False


def test_basic_tools():
    """Test individual MCP tools with sample data."""
    if not MCP_AVAILABLE:
        print("Skipping MCP tests - not available")
        return

    sample_text = """
    Name: Test Student
    Date: 2025-01-15
    Class: 10
    Subject: English
    
    The Renaissance was a period of great cultural and artistic achievement in Europe. 
    This era saw the development of new artistic techniques and scientific discoveries.
    Many famous artists like Leonardo da Vinci and Michelangelo created masterpieces during this time.
    """

    source_text = "The Renaissance was a cultural movement that spanned the 14th to 17th centuries, beginning in Italy and later spreading to the rest of Europe."

    print("\nTesting MCP Tools")
    print("=" * 30)

    # Test grammar check
    print("\n1. Testing Grammar Check:")
    try:
        result = grammar_check(sample_text)
        print(f"   Status: {result.get('status')}")
        print(f"   Grammar errors: {result.get('grammar_errors')}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test summarization
    print("\n2. Testing Summarization:")
    try:
        result = summarize_assignment(sample_text)
        print(f"   Status: {result.get('status')}")
        if result.get("summary"):
            print(f"   Summary: {result['summary'][:100]}...")
    except Exception as e:
        print(f"   Error: {e}")

    # Test relevance check
    print("\n3. Testing Relevance Check:")
    try:
        result = relevance_check(sample_text, source_text)
        print(f"   Status: {result.get('status')}")
        if result.get("relevance_analysis"):
            print(f"   Analysis: {result['relevance_analysis'][:100]}...")
    except Exception as e:
        print(f"   Error: {e}")

    # Test plagiarism check
    print("\n4. Testing Plagiarism Check:")
    try:
        result = plagiarism_check(sample_text, "Test_Student")
        print(f"   Status: {result.get('status')}")
        print(f"   Report file: {result.get('report_file')}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test grading
    print("\n5. Testing Grading:")
    try:
        result = grade_assignment(sample_text, source_text)
        print(f"   Status: {result.get('status')}")
        if result.get("grades"):
            grades = result["grades"]
            print(
                f"   Grades: Factuality={grades.get('factuality')}, Relevance={grades.get('relevance')}, Coherence={grades.get('coherence')}, Grammar={grades.get('grammar')}"
            )
    except Exception as e:
        print(f"   Error: {e}")


async def test_parallel_processing():
    """Test the parallel processing tool."""
    if not MCP_AVAILABLE:
        print("Skipping parallel processing test - MCP not available")
        return

    print("\nTesting Parallel Processing")
    print("=" * 35)

    sample_text = """
    Name: Parallel Test Student
    Date: 2025-01-15
    Class: 10
    Subject: History
    
    The Renaissance period marked a significant transformation in European culture.
    During this era, there was a renewed interest in classical learning and humanism.
    Artists and scholars made groundbreaking contributions to art, science, and literature.
    This period laid the foundation for the modern world we know today.
    """

    source_text = "The Renaissance was a cultural movement that spanned the 14th to 17th centuries, beginning in Italy and later spreading to the rest of Europe."

    try:
        result = await process_assignment_parallel(
            sample_text, source_text, "Parallel_Test_Student"
        )

        print(f"Processing Status: {result.get('processing_status')}")
        print(f"Student: {result.get('student_name')}")

        # Check individual tool results
        tools = [
            "grammar_check",
            "plagiarism_check",
            "relevance_check",
            "grading",
            "summary",
        ]
        for tool in tools:
            tool_result = result.get(tool, {})
            status = tool_result.get("status", "unknown")
            print(f"   {tool}: {status}")

    except Exception as e:
        print(f"Parallel processing error: {e}")


def test_environment():
    """Test environment setup."""
    print("Environment Check")
    print("=" * 20)

    # Check API key
    groq_key = os.getenv("GROQ_API_KEY")
    print(f"GROQ_API_KEY: {'Set' if groq_key else 'Missing'}")

    # Check LLM initialization
    try:
        from llms import groq_llm

        print(f"Groq LLM: {'Initialized' if groq_llm else 'Failed to initialize'}")
    except Exception as e:
        print(f"Groq LLM: Error - {e}")

    # Check required directories
    from paths import ASSIGNMENTS_FOLDER, PLAGIARISM_REPORTS_FOLDER

    print(
        f"Assignments folder: {ASSIGNMENTS_FOLDER} ({'exists' if os.path.exists(ASSIGNMENTS_FOLDER) else 'missing'})"
    )
    print(
        f"Reports folder: {PLAGIARISM_REPORTS_FOLDER} ({'exists' if os.path.exists(PLAGIARISM_REPORTS_FOLDER) else 'missing'})"
    )


async def main():
    """Run all tests."""
    print("MCP Tools Test Suite")
    print("=" * 25)

    # Environment check
    test_environment()

    # Basic tools test
    test_basic_tools()

    # Parallel processing test
    await test_parallel_processing()

    print("\nTest suite completed!")


if __name__ == "__main__":
    asyncio.run(main())
