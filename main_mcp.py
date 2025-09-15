"""
Updated main entry point with MCP integration support.
This version provides both traditional processing and MCP server capabilities.
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
            "Initial Grade": result.get("grading", {}).get("grades", "N/A")
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
            "Initial Grade": "N/A"
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
            "Initial Grade": final_state["grade"]
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
            "Initial Grade": "N/A"
        }


@traceable(name="process_assignments_batch")
async def process_assignments_batch(use_mcp: bool = True) -> List[Dict]:
    """Process all assignments in the assignments folder."""
    print(f"Processing assignments with {'MCP tools' if use_mcp else 'traditional processing'}...")
    ensure_directories()
    
    if not ASSIGNMENTS_FOLDER or not os.path.isdir(ASSIGNMENTS_FOLDER):
        raise ValueError("ASSIGNMENTS_FOLDER is not set or is invalid.")

    source_text = """The Renaissance was a cultural movement..."""
    assignments = []

    for file in os.listdir(ASSIGNMENTS_FOLDER):
        if file.endswith(".txt"):
            file_path = os.path.join(ASSIGNMENTS_FOLDER, file)
            
            if use_mcp and MCP_AVAILABLE:
                result = await process_assignment_with_mcp(file_path, source_text)
            else:
                result = await process_assignment_traditional(file_path, source_text)
            
            assignments.append(result)

    # Export results using traditional export function
    from nodes import export_summary
    csv_path = export_summary(assignments, SUMMARY_CSV_PATH)
    print(f"[INFO] Summary exported to {csv_path}")
    return assignments


async def run_mcp_server():
    """Run the MCP server for external tool access."""
    if not MCP_AVAILABLE:
        print("[ERROR] MCP is not available. Please install with: pip install 'mcp[cli]'")
        return
    
    from mcp_server import mcp
    print("🚀 Starting MCP server...")
    print("Available tools: grammar_check, plagiarism_check, relevance_check, grade_assignment, summarize_assignment, process_assignment_parallel")
    mcp.run(transport="stdio")


async def main():
    """Main entry point with mode selection."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "mcp":
            # Run as MCP server
            await run_mcp_server()
        elif mode == "traditional":
            # Force traditional processing
            await process_assignments_batch(use_mcp=False)
        elif mode == "help":
            print_help()
        else:
            print(f"Unknown mode: {mode}")
            print_help()
    else:
        # Default: batch processing with MCP if available
        await process_assignments_batch(use_mcp=MCP_AVAILABLE)


def print_help():
    """Print help information."""
    print("Assignment Grading System - MCP Enhanced")
    print("=" * 45)
    print("Usage:")
    print("  python main_mcp.py              # Process assignments (MCP if available)")
    print("  python main_mcp.py mcp          # Run as MCP server")
    print("  python main_mcp.py traditional  # Force traditional processing")
    print("  python main_mcp.py help         # Show this help")
    print("")
    print("MCP Status:", "✅ Available" if MCP_AVAILABLE else "❌ Not available")
    print("LangSmith Status:", "✅ Available" if LANGSMITH_AVAILABLE else "❌ Not available")


if __name__ == '__main__':
    asyncio.run(main())