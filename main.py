import asyncio
import os
from typing import Dict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph

from nodes import (
    export_summary,
    orchestrator_node
)
from paths import ASSIGNMENTS_FOLDER, SUMMARY_CSV_PATH
from utils import ensure_directories, extract_metadata_from_content, graph_visualizer

load_dotenv()


def build_graph() -> StateGraph:
    """Build and configure the processing graph with orchestrator for parallel execution."""
    graph = StateGraph(dict)
    graph.add_node("Orchestrator", orchestrator_node)
    graph.set_entry_point("Orchestrator")
    return graph.compile()


async def run_pipeline_file(graph, file_path: str, source_text: str) -> Dict:
    """Process a single assignment file through the pipeline."""
    with open(file_path, "r") as f:
        content = f.read()

    metadata = extract_metadata_from_content(file_path)

    initial_state = {
        "content": content,
        "metadata": metadata,
        "source_text": source_text
    }

    try:
        final_state = await graph.ainvoke(initial_state)
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
        print(f"[ERROR] Graph execution failed for {file_path}: {e}")
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


async def process_assignments() -> List[Dict]:
    """Process all assignments in the assignments folder."""
    print("Processing assignments...")
    ensure_directories()
    
    if not ASSIGNMENTS_FOLDER or not os.path.isdir(ASSIGNMENTS_FOLDER):
        raise ValueError("ASSIGNMENTS_FOLDER is not set or is invalid.")

    source_text = """The Renaissance was a cultural movement..."""
    assignments = []
    graph = build_graph()

    for file in os.listdir(ASSIGNMENTS_FOLDER):
        if file.endswith(".txt"):
            file_path = os.path.join(ASSIGNMENTS_FOLDER, file)
            result = await run_pipeline_file(graph, file_path, source_text)
            assignments.append(result)

    csv_path = export_summary(assignments, SUMMARY_CSV_PATH)
    print(f"[INFO] Summary exported to {csv_path}")
    return assignments


async def main():
    """Main entry point for the application."""
    graph = build_graph()
    graph_visualizer(graph)
    await process_assignments()


if __name__ == '__main__':
    asyncio.run(main())