import os
from typing import Dict

from core.paths import REQUIRED_FOLDERS


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    for folder in REQUIRED_FOLDERS:
        os.makedirs(folder, exist_ok=True)


def extract_metadata_from_content(
    file_path: str, content: str = None
) -> Dict[str, str]:
    """Extract metadata from assignment file header.

    Expected format:
    Name: John Doe
    Date: 2025-08-25
    Class: 10
    Subject: English

    Args:
        file_path: Path to the file (for filename fallback)
        content: Optional content string (if already extracted)
    """
    print("Extracting metadata from content...")

    if content is None:
        # Read from file if content not provided
        try:
            with open(file_path, "r") as f:
                lines = [line.strip() for line in f.readlines()[:10]]
        except Exception:
            # If file reading fails, use filename as fallback
            import os

            filename = os.path.splitext(os.path.basename(file_path))[0]
            return {
                "name": filename,
                "date": "Unknown",
                "class": "Unknown",
                "subject": "Unknown",
            }
    else:
        # Extract from provided content
        lines = [line.strip() for line in content.split("\n")[:10]]

    meta = {
        "name": "Unknown",
        "date": "Unknown",
        "class": "Unknown",
        "subject": "Unknown",
    }

    for line in lines:
        if line.lower().startswith("name:"):
            meta["name"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("date:"):
            meta["date"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("class:"):
            meta["class"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("subject:"):
            meta["subject"] = line.split(":", 1)[1].strip()

    # If name is still unknown, try to extract from filename
    if meta["name"] == "Unknown" and file_path:
        import os

        filename = os.path.splitext(os.path.basename(file_path))[0]
        # Clean up filename to extract potential student name
        meta["name"] = filename.replace("_", " ").replace("-", " ").title()

    return meta


def graph_visualizer(graph) -> None:
    """Generate and save graph visualization as PNG with async node details."""
    print("Generating graph...")
    try:
        from core.paths import GRAPH_OUTPUT_PATH

        # Generate the main graph visualization
        image_data = graph.get_graph().draw_mermaid_png()
        with open(GRAPH_OUTPUT_PATH, "wb") as f:
            f.write(image_data)
        print(f"Graph saved as '{GRAPH_OUTPUT_PATH}'. Open it to view the structure.")

        # Print detailed async execution info to console
        print("\nGraph Structure:")
        print("|-- Orchestrator Node (ASYNC)")
        print("|   |-- Executes in parallel:")
        print("|       |-- Grammar Check Node")
        print("|       |-- Plagiarism Check Node")
        print("|       |-- Source Check Node")
        print("|       |-- Initial Grading Node")
        print("|       |-- Summary Node")
        print("|-- All nodes run concurrently using asyncio.gather()\n")

    except Exception as e:
        print(f"Error generating graph: {e}")
        # Fallback: still show the structure info even if visualization fails
        print("\nGraph Structure (Visualization failed):")
        print("|-- Orchestrator Node (ASYNC)")
        print("|   |-- Executes in parallel:")
        print("|       |-- Grammar Check Node")
        print("|       |-- Plagiarism Check Node")
        print("|       |-- Source Check Node")
        print("|       |-- Initial Grading Node")
        print("|       |-- Summary Node")
        print("|-- All nodes run concurrently using asyncio.gather()\n")
