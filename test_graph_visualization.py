#!/usr/bin/env python3
"""
Test script to visualize the LangGraph workflow.
Run this script to see the current graph structure.
"""

from visualize_graph import (
    visualize_workflow_graph,
    print_workflow_structure,
    get_current_graph_json
)
from simple_graph_viz import visualize_simple_workflow
import json


def main():
    """Main test function to demonstrate graph visualization."""
    print("🚀 Testing LangGraph Workflow Visualization")
    print("=" * 50)

    # Option 1: Print text-based structure
    print("\n1️⃣ Text-based workflow structure:")
    print_workflow_structure()

    # Option 2: Create simplified visual graph
    print("\n2️⃣ Creating simplified visual graph...")
    try:
        result = visualize_simple_workflow(
            save_path="simple_workflow_test.png",
            show_plot=False
        )
        print("✅ Simplified visual graph created successfully!")
        print(result)
    except Exception as e:
        print(f"❌ Error creating simplified graph: {e}")

    # Option 3: Create detailed visual graph (with all conditional edges)
    print("\n3️⃣ Creating detailed visual graph...")
    try:
        summary = visualize_workflow_graph(
            save_path="detailed_workflow_test.png",
            show_plot=False
        )
        print("✅ Detailed visual graph created successfully!")
    except Exception as e:
        print(f"❌ Error creating detailed graph: {e}")
        print("Note: Make sure matplotlib and networkx are installed")

    # Option 4: Get JSON representation
    print("\n4️⃣ JSON structure of the workflow:")
    try:
        graph_json = get_current_graph_json()
        print(json.dumps(graph_json["workflow_metadata"], indent=2))

        print(f"\nMain workflow nodes:")
        main_nodes = [n for n in graph_json['nodes'] if not n['id'].startswith('__')]
        for node in main_nodes:
            print(f"  • {node['name']} (ID: {node['id']})")

    except Exception as e:
        print(f"❌ Error getting JSON structure: {e}")

    print("\n" + "=" * 50)
    print("🎯 Test completed!")
    print("\nGenerated files:")
    print("  • simple_workflow_test.png - Clean, readable workflow view")
    print("  • detailed_workflow_test.png - Complete graph with all edges")
    print("\nAvailable functions:")
    print("  • visualize_simple_workflow() - Clean workflow view")
    print("  • visualize_workflow_graph() - Detailed graph view")
    print("  • print_workflow_structure() - Text structure")
    print("  • get_current_graph_json() - JSON representation")


if __name__ == "__main__":
    main()