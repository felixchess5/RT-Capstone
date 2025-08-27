import os
from typing import Dict

from paths import REQUIRED_FOLDERS


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    for folder in REQUIRED_FOLDERS:
        os.makedirs(folder, exist_ok=True)


def extract_metadata_from_content(file_path: str) -> Dict[str, str]:
    """Extract metadata from assignment file header.
    
    Expected format:
    Name: John Doe
    Date: 2025-08-25
    Class: 10
    Subject: English
    """
    print("Extracting metadata from content...")
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()[:10]]

    meta = {"name": "Unknown", "date": "Unknown", "class": "Unknown", "subject": "Unknown"}
    
    for line in lines:
        if line.lower().startswith("name:"):
            meta["name"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("date:"):
            meta["date"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("class:"):
            meta["class"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("subject:"):
            meta["subject"] = line.split(":", 1)[1].strip()

    return meta


def graph_visualizer(graph) -> None:
    """Generate and save graph visualization as PNG with async node details."""
    print("Generating graph...")
    try:
        from paths import GRAPH_OUTPUT_PATH
        
        # Generate the main graph visualization
        image_data = graph.get_graph().draw_mermaid_png()
        with open(GRAPH_OUTPUT_PATH, "wb") as f:
            f.write(image_data)
        print(f"✅ Graph saved as '{GRAPH_OUTPUT_PATH}'. Open it to view the structure.")
        
        # Print detailed async execution info to console
        print("\n📊 Graph Structure:")
        print("├── Orchestrator Node (ASYNC)")
        print("│   └── Executes in parallel:")
        print("│       ├── Grammar Check Node")
        print("│       ├── Plagiarism Check Node") 
        print("│       ├── Source Check Node")
        print("│       ├── Initial Grading Node")
        print("│       └── Summary Node")
        print("└── All nodes run concurrently using asyncio.gather()\n")
        
    except Exception as e:
        print(f"Error generating graph: {e}")
        # Fallback: still show the structure info even if visualization fails
        print("\n📊 Graph Structure (Visualization failed):")
        print("├── Orchestrator Node (ASYNC)")
        print("│   └── Executes in parallel:")
        print("│       ├── Grammar Check Node")
        print("│       ├── Plagiarism Check Node") 
        print("│       ├── Source Check Node")
        print("│       ├── Initial Grading Node")
        print("│       └── Summary Node")
        print("└── All nodes run concurrently using asyncio.gather()\n")