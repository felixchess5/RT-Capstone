"""
Main entry point with Agentic AI Workflow integration.
This version provides traditional processing, MCP server capabilities, and the new agentic workflow.
"""
import asyncio
import csv
import os
import sys
from typing import Dict, List
from dotenv import load_dotenv

try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    def traceable(func=None, **kwargs):
        def decorator(f):
            return f
        return decorator(func) if func else decorator
    LANGSMITH_AVAILABLE = False

# Try to import agentic workflow
try:
    from agentic_workflow import run_agentic_workflow, build_agentic_workflow
    AGENTIC_AVAILABLE = True
    print("✅ Agentic workflow loaded successfully")
except ImportError as e:
    AGENTIC_AVAILABLE = False
    print(f"⚠️ Agentic workflow not available: {e}")

# Try to import MCP tools
try:
    from mcp_server import (
        grammar_check,
        plagiarism_check,
        relevance_check,
        grade_assignment,
        summarize_assignment,
        process_assignment_parallel
    )
    MCP_AVAILABLE = True
    print("✅ MCP tools loaded successfully")
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP tools not available, falling back to traditional processing")
    # Import traditional nodes as fallback
    from nodes import (
        export_summary,
        grammar_check_fn,
        plagiarism_check_fn,
        relevance_check,
        grading_node,
        summarize
    )

from paths import ASSIGNMENTS_FOLDER, SUMMARY_CSV_PATH
from utils import ensure_directories, extract_metadata_from_content

load_dotenv()


@traceable(name="process_assignment_agentic")
async def process_assignment_agentic(file_path: str, source_text: str) -> Dict:
    """Process a single assignment using the agentic AI workflow."""
    with open(file_path, "r") as f:
        content = f.read()

    metadata = extract_metadata_from_content(file_path)

    try:
        # Use the agentic workflow
        result = await run_agentic_workflow(content, metadata, source_text)

        # Transform agentic results to match expected CSV format
        return {
            "Student Name": result.get("student_name", metadata["name"]),
            "Date of Submission": result.get("date_of_submission", metadata["date"]),
            "Class": result.get("class", metadata["class"]),
            "Subject": result.get("subject", metadata["subject"]),
            "Summary": result.get("summary", "N/A"),
            "Grammar Errors": result.get("grammar_errors", "N/A"),
            "Plagiarism File": result.get("plagiarism_file", "N/A"),
            "Content Relevance": result.get("content_relevance", "N/A"),
            "Initial Grade": result.get("initial_grade", "N/A"),
            "Overall Score": result.get("overall_score", "N/A"),
            "Letter Grade": result.get("letter_grade", "N/A"),
            "Workflow Version": result.get("workflow_version", "agentic_v1")
        }
    except Exception as e:
        print(f"[ERROR] Agentic processing failed for {file_path}: {e}")
        return {
            "Student Name": metadata["name"],
            "Date of Submission": metadata["date"],
            "Class": metadata["class"],
            "Subject": metadata["subject"],
            "Summary": f"Agentic Error: {str(e)}",
            "Grammar Errors": "N/A",
            "Plagiarism File": "N/A",
            "Content Relevance": "N/A",
            "Initial Grade": "N/A",
            "Overall Score": "N/A",
            "Letter Grade": "N/A",
            "Workflow Version": "error"
        }


@traceable(name="process_assignment_with_mcp")
async def process_assignment_with_mcp(file_path: str, source_text: str) -> Dict:
    """Process a single assignment using MCP tools."""
    with open(file_path, "r") as f:
        content = f.read()

    metadata = extract_metadata_from_content(file_path)
    student_name = metadata["name"]

    try:
        # Use the parallel processing MCP tool
        result = await process_assignment_parallel(content, source_text, student_name)

        # Transform MCP results to match expected format
        return {
            "Student Name": metadata["name"],
            "Date of Submission": metadata["date"],
            "Class": metadata["class"],
            "Subject": metadata["subject"],
            "Summary": result.get("summary", {}).get("summary", "N/A"),
            "Grammar Errors": result.get("grammar_check", {}).get("grammar_errors", "N/A"),
            "Plagiarism File": result.get("plagiarism_check", {}).get("report_file", "N/A"),
            "Content Relevance": result.get("relevance_check", {}).get("relevance_analysis", "N/A"),
            "Initial Grade": result.get("grading", {}).get("grades", "N/A"),
            "Overall Score": "N/A",
            "Letter Grade": "N/A",
            "Workflow Version": "mcp_v1"
        }
    except Exception as e:
        print(f"[ERROR] MCP processing failed for {file_path}: {e}")
        return {
            "Student Name": metadata["name"],
            "Date of Submission": metadata["date"],
            "Class": metadata["class"],
            "Subject": metadata["subject"],
            "Summary": f"MCP Error: {str(e)}",
            "Grammar Errors": "N/A",
            "Plagiarism File": "N/A",
            "Content Relevance": "N/A",
            "Initial Grade": "N/A",
            "Overall Score": "N/A",
            "Letter Grade": "N/A",
            "Workflow Version": "error"
        }


@traceable(name="process_assignment_traditional")
async def process_assignment_traditional(file_path: str, source_text: str) -> Dict:
    """Process a single assignment using traditional nodes."""
    from nodes import orchestrator_node

    with open(file_path, "r") as f:
        content = f.read()

    metadata = extract_metadata_from_content(file_path)

    initial_state = {
        "content": content,
        "metadata": metadata,
        "source_text": source_text
    }

    try:
        final_state = await orchestrator_node(initial_state)
        return {
            "Student Name": final_state["metadata"]["name"],
            "Date of Submission": final_state["metadata"]["date"],
            "Class": final_state["metadata"]["class"],
            "Subject": final_state["metadata"]["subject"],
            "Summary": final_state["summary"],
            "Grammar Errors": final_state["grammar_errors"],
            "Plagiarism File": final_state["plagiarism_file"],
            "Content Relevance": final_state["relevance"],
            "Initial Grade": final_state["grade"],
            "Overall Score": "N/A",
            "Letter Grade": "N/A",
            "Workflow Version": "traditional_v1"
        }
    except Exception as e:
        print(f"[ERROR] Traditional processing failed for {file_path}: {e}")
        return {
            "Student Name": metadata["name"],
            "Date of Submission": metadata["date"],
            "Class": metadata["class"],
            "Subject": metadata["subject"],
            "Summary": f"Error: {str(e)}",
            "Grammar Errors": "N/A",
            "Plagiarism File": "N/A",
            "Content Relevance": "N/A",
            "Initial Grade": "N/A",
            "Overall Score": "N/A",
            "Letter Grade": "N/A",
            "Workflow Version": "error"
        }


@traceable(name="process_assignments_batch")
async def process_assignments_batch(processing_mode: str = "auto") -> List[Dict]:
    """Process all assignments in the assignments folder."""
    print(f"Processing assignments with {processing_mode} mode...")
    ensure_directories()

    if not ASSIGNMENTS_FOLDER or not os.path.isdir(ASSIGNMENTS_FOLDER):
        raise ValueError("ASSIGNMENTS_FOLDER is not set or is invalid.")

    source_text = """The Renaissance was a cultural movement that spanned roughly the 14th to the 17th century, beginning in Italy in the Late Middle Ages and later spreading to the rest of Europe. The term is also used more loosely to refer to the historical era, but since the changes of the Renaissance were not uniform across Europe, this is a general use of the term. As a cultural movement, it encompassed innovative flowering of Latin and vernacular literatures, beginning with the 14th-century resurgence of learning based on classical sources, which contemporaries credited to Petrarch, the development of linear perspective and other techniques of rendering a more natural reality in painting, and gradual but widespread educational reform."""

    assignments = []

    # Determine processing function based on mode and availability
    if processing_mode == "agentic" and AGENTIC_AVAILABLE:
        process_func = process_assignment_agentic
        print("🤖 Using Agentic AI Workflow")
    elif processing_mode == "mcp" and MCP_AVAILABLE:
        process_func = process_assignment_with_mcp
        print("🔧 Using MCP Tools")
    elif processing_mode == "traditional":
        process_func = process_assignment_traditional
        print("📊 Using Traditional Processing")
    elif processing_mode == "auto":
        # Auto-select the best available method
        if AGENTIC_AVAILABLE:
            process_func = process_assignment_agentic
            print("🤖 Auto-selected: Agentic AI Workflow")
        elif MCP_AVAILABLE:
            process_func = process_assignment_with_mcp
            print("🔧 Auto-selected: MCP Tools")
        else:
            process_func = process_assignment_traditional
            print("📊 Auto-selected: Traditional Processing")
    else:
        # Fallback to traditional
        process_func = process_assignment_traditional
        print("📊 Falling back to Traditional Processing")

    for file in os.listdir(ASSIGNMENTS_FOLDER):
        if file.endswith(".txt"):
            file_path = os.path.join(ASSIGNMENTS_FOLDER, file)
            print(f"Processing: {file}")
            result = await process_func(file_path, source_text)
            assignments.append(result)

    # Export results with enhanced CSV format
    csv_path = export_enhanced_summary(assignments, SUMMARY_CSV_PATH)
    print(f"[INFO] Enhanced summary exported to {csv_path}")
    return assignments


def export_enhanced_summary(assignments: List[Dict], output_path: str) -> str:
    """Export assignment summaries to CSV file with enhanced fields."""
    print("Exporting enhanced summary...")

    # Define enhanced fieldnames
    fieldnames = [
        "Student Name", "Date of Submission", "Class", "Subject", "Summary",
        "Grammar Errors", "Plagiarism File", "Content Relevance", "Initial Grade",
        "Overall Score", "Letter Grade", "Workflow Version"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for assignment in assignments:
            # Ensure all fields are present
            row = {field: assignment.get(field, "N/A") for field in fieldnames}
            writer.writerow(row)
    return output_path


async def run_mcp_server():
    """Run the MCP server for external tool access."""
    if not MCP_AVAILABLE:
        print("[ERROR] MCP is not available. Please install with: pip install 'mcp[cli]'")
        return

    from mcp_server import mcp
    print("🚀 Starting MCP server...")
    print("Available tools: grammar_check, plagiarism_check, relevance_check, grade_assignment, summarize_assignment, process_assignment_parallel")
    mcp.run(transport="stdio")


async def run_workflow_demo():
    """Run a demonstration of the agentic workflow."""
    if not AGENTIC_AVAILABLE:
        print("[ERROR] Agentic workflow is not available.")
        return

    print("🚀 Running Agentic Workflow Demo...")

    # Create a demo workflow graph
    workflow = build_agentic_workflow()

    # Generate workflow visualization if possible
    try:
        from utils import graph_visualizer
        graph_visualizer(workflow, "agentic_workflow_demo")
        print("📊 Workflow visualization generated")
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    print("✅ Demo workflow built successfully!")
    print("Use 'python main_agentic.py agentic' to process assignments with the agentic workflow")


async def main():
    """Main entry point with enhanced mode selection."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "mcp":
            # Run as MCP server
            await run_mcp_server()
        elif mode == "agentic":
            # Use agentic workflow
            await process_assignments_batch(processing_mode="agentic")
        elif mode == "traditional":
            # Force traditional processing
            await process_assignments_batch(processing_mode="traditional")
        elif mode == "demo":
            # Run workflow demonstration
            await run_workflow_demo()
        elif mode == "compare":
            # Run comparison between different methods
            await run_comparison_analysis()
        elif mode == "help":
            print_help()
        else:
            print(f"Unknown mode: {mode}")
            print_help()
    else:
        # Default: auto-select best available processing method
        await process_assignments_batch(processing_mode="auto")


async def run_comparison_analysis():
    """Run assignments through different processing methods for comparison."""
    print("🔄 Running comparison analysis across all processing methods...")

    ensure_directories()

    if not ASSIGNMENTS_FOLDER or not os.path.isdir(ASSIGNMENTS_FOLDER):
        raise ValueError("ASSIGNMENTS_FOLDER is not set or is invalid.")

    files = [f for f in os.listdir(ASSIGNMENTS_FOLDER) if f.endswith(".txt")]

    if not files:
        print("No assignment files found for comparison.")
        return

    # Take the first file for comparison
    test_file = files[0]
    file_path = os.path.join(ASSIGNMENTS_FOLDER, test_file)
    source_text = "The Renaissance was a cultural movement..."

    print(f"📋 Comparing processing methods on: {test_file}")

    results = {}

    # Test agentic workflow
    if AGENTIC_AVAILABLE:
        print("🤖 Testing Agentic Workflow...")
        start_time = asyncio.get_event_loop().time()
        agentic_result = await process_assignment_agentic(file_path, source_text)
        agentic_time = asyncio.get_event_loop().time() - start_time
        results["agentic"] = {"result": agentic_result, "time": agentic_time}

    # Test MCP tools
    if MCP_AVAILABLE:
        print("🔧 Testing MCP Tools...")
        start_time = asyncio.get_event_loop().time()
        mcp_result = await process_assignment_with_mcp(file_path, source_text)
        mcp_time = asyncio.get_event_loop().time() - start_time
        results["mcp"] = {"result": mcp_result, "time": mcp_time}

    # Test traditional processing
    print("📊 Testing Traditional Processing...")
    start_time = asyncio.get_event_loop().time()
    traditional_result = await process_assignment_traditional(file_path, source_text)
    traditional_time = asyncio.get_event_loop().time() - start_time
    results["traditional"] = {"result": traditional_result, "time": traditional_time}

    # Print comparison results
    print("\n" + "="*60)
    print("PROCESSING METHOD COMPARISON")
    print("="*60)

    for method, data in results.items():
        print(f"\n{method.upper()} METHOD:")
        print(f"  Processing Time: {data['time']:.2f} seconds")
        print(f"  Summary: {data['result']['Summary'][:100]}...")
        print(f"  Grammar Errors: {data['result']['Grammar Errors']}")
        print(f"  Initial Grade: {data['result']['Initial Grade']}")
        if 'Overall Score' in data['result']:
            print(f"  Overall Score: {data['result']['Overall Score']}")
        if 'Letter Grade' in data['result']:
            print(f"  Letter Grade: {data['result']['Letter Grade']}")


def print_help():
    """Print help information."""
    print("Assignment Grading System - Enhanced with Agentic AI Workflow")
    print("=" * 65)
    print("Usage:")
    print("  python main_agentic.py              # Auto-select best processing method")
    print("  python main_agentic.py agentic      # Use agentic AI workflow")
    print("  python main_agentic.py mcp          # Run as MCP server")
    print("  python main_agentic.py traditional  # Force traditional processing")
    print("  python main_agentic.py demo         # Demo agentic workflow")
    print("  python main_agentic.py compare      # Compare all processing methods")
    print("  python main_agentic.py help         # Show this help")
    print("")
    print("System Status:")
    print("  Agentic Workflow:", "✅ Available" if AGENTIC_AVAILABLE else "❌ Not available")
    print("  MCP Tools:", "✅ Available" if MCP_AVAILABLE else "❌ Not available")
    print("  LangSmith:", "✅ Available" if LANGSMITH_AVAILABLE else "❌ Not available")
    print("")
    print("Features:")
    print("  🤖 Agentic AI Workflow - Full state machine with error recovery")
    print("  🔧 MCP Tools - Model Context Protocol integration")
    print("  📊 Traditional Processing - Original async orchestration")
    print("  🔄 Auto-Selection - Intelligent method selection")
    print("  📈 Comparison Mode - Side-by-side method comparison")


if __name__ == '__main__':
    asyncio.run(main())