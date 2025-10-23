"""
Graph visualization utility for the LangGraph agentic workflow.
Provides functions to visualize the current workflow structure.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from typing import Dict, List, Tuple
import json
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from workflows.agentic_workflow import build_agentic_workflow, WorkflowStep


def get_workflow_graph_info():
    """
    Extract graph structure information from the compiled workflow.
    Returns nodes, edges, and metadata about the workflow.
    """
    # Build the workflow to get its structure
    workflow = build_agentic_workflow()

    # Extract nodes and edges information
    nodes = []
    edges = []

    # Get the underlying graph structure
    # LangGraph stores the graph structure internally
    graph_dict = workflow.get_graph().to_json()

    # Debug: print the structure to understand the format
    print(f"🔍 Graph structure debug:")
    print(f"   Nodes type: {type(graph_dict['nodes'])}")
    print(f"   Nodes sample: {list(graph_dict['nodes'])[:5] if isinstance(graph_dict['nodes'], (list, dict)) else str(graph_dict['nodes'])[:100]}")
    print(f"   Edges type: {type(graph_dict['edges'])}")
    print(f"   Edges count: {len(graph_dict['edges']) if hasattr(graph_dict['edges'], '__len__') else 'unknown'}")

    # Parse nodes - handle both dict and list formats
    if isinstance(graph_dict["nodes"], dict):
        for node_id, node_data in graph_dict["nodes"].items():
            nodes.append({
                "id": node_id,
                "type": node_data.get("type", "node") if isinstance(node_data, dict) else "node",
                "name": node_id.replace("_", " ").title()
            })
    else:
        # Handle list format
        for node in graph_dict["nodes"]:
            if isinstance(node, dict):
                node_id = node.get("id", str(node))
                nodes.append({
                    "id": node_id,
                    "type": node.get("type", "node"),
                    "name": node_id.replace("_", " ").title()
                })
            else:
                nodes.append({
                    "id": str(node),
                    "type": "node",
                    "name": str(node).replace("_", " ").title()
                })

    # Parse edges - be more careful about format
    edge_count = 0
    for edge in graph_dict["edges"]:
        edge_count += 1
        if edge_count <= 5:  # Debug first few edges
            print(f"   Edge {edge_count}: {edge} (type: {type(edge)})")

        if isinstance(edge, dict):
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                edges.append((source, target))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            edges.append((edge[0], edge[1]))
        else:
            # Try to parse as string or other format
            try:
                edge_str = str(edge)
                if " -> " in edge_str:
                    parts = edge_str.split(" -> ")
                    if len(parts) == 2:
                        edges.append((parts[0].strip(), parts[1].strip()))
            except:
                pass

    print(f"   Parsed {len(nodes)} nodes and {len(edges)} edges")
    return nodes, edges, graph_dict


def visualize_workflow_graph(save_path: str = None, show_plot: bool = True) -> str:
    """
    Create a visual representation of the LangGraph workflow.

    Args:
        save_path: Optional path to save the visualization
        show_plot: Whether to display the plot

    Returns:
        Information about the graph structure
    """
    print("🎨 Creating workflow visualization...")

    # Get graph information
    nodes, edges, graph_dict = get_workflow_graph_info()

    # Create NetworkX graph for layout
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in nodes:
        G.add_node(node["id"], **node)

    # Add edges
    G.add_edges_from(edges)

    # Create the plot
    plt.figure(figsize=(16, 12))

    # Use hierarchical layout for better workflow visualization
    if len(G.nodes()) == 0:
        print("❌ No nodes found in graph - cannot create visualization")
        return "Error: No nodes found"

    try:
        # Try different layout algorithms
        if len(G.nodes()) > 10:
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        else:
            pos = nx.circular_layout(G)
    except Exception as e:
        print(f"⚠️ Layout algorithm failed: {e}, using fallback")
        pos = {node: (i % 4, i // 4) for i, node in enumerate(G.nodes())}

    # Define colors for different node types
    node_colors = {
        'initialize': '#4CAF50',  # Green for start
        'quality_check': '#2196F3',  # Blue for analysis
        'subject_classification': '#FF9800',  # Orange for classification
        'specialized_processing': '#9C27B0',  # Purple for specialized
        'grammar_analysis': '#2196F3',  # Blue for analysis
        'plagiarism_detection': '#F44336',  # Red for detection
        'relevance_analysis': '#2196F3',  # Blue for analysis
        'content_grading': '#4CAF50',  # Green for grading
        'summary_generation': '#795548',  # Brown for summary
        'quality_validation': '#607D8B',  # Blue Grey for validation
        'error_recovery': '#FF5722',  # Deep Orange for recovery
        'results_aggregation': '#3F51B5',  # Indigo for aggregation
        'finalize': '#9E9E9E'  # Grey for end
    }

    # Create labels
    labels = {}
    for node in G.nodes():
        label = node.replace('_', '\n').title()
        # Shorten very long labels
        if len(label) > 20:
            lines = label.split('\n')
            if len(lines) > 2:
                labels[node] = '\n'.join(lines[:2]) + '...'
            else:
                labels[node] = label[:15] + '...' if len(label) > 15 else label
        else:
            labels[node] = label

    # Get node colors
    node_list = list(G.nodes())
    colors = [node_colors.get(node, '#E0E0E0') for node in node_list]

    # Draw the complete graph
    nx.draw(G, pos,
            node_color=colors,
            node_size=1500,
            with_labels=True,
            labels=labels,
            font_size=7,
            font_weight='bold',
            edge_color='gray',
            width=0.5,
            alpha=0.8,
            arrows=True,
            arrowsize=15)

    # Add title and labels
    plt.title('LangGraph Agentic Workflow Structure',
              fontsize=16, fontweight='bold', pad=20)

    # Create legend
    legend_elements = []
    for node_type, color in node_colors.items():
        legend_elements.append(mpatches.Patch(color=color,
                                            label=node_type.replace('_', ' ').title()))

    plt.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1, 0.5),
              fontsize=10)

    # Remove axes
    plt.axis('off')

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Save if requested
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📁 Graph saved to: {save_path}")
        except Exception as e:
            print(f"❌ Failed to save graph: {e}")

    # Show if requested
    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"⚠️ Cannot display plot (no GUI available): {e}")
            print("Graph was saved to file instead")

    # Return summary information
    summary = f"""
Workflow Graph Summary:
- Total Nodes: {len(nodes)}
- Total Edges: {len(edges)}
- Entry Point: initialize
- Exit Point: finalize

Node Types:
{chr(10).join([f"  • {node['name']}" for node in nodes])}

Workflow Flow:
The workflow follows a conditional routing pattern where each step
determines the next step based on processing requirements and results.
    """

    print(summary)
    return summary


def print_workflow_structure():
    """
    Print a text-based representation of the workflow structure.
    """
    print("📋 LangGraph Workflow Structure")
    print("=" * 50)

    # Get workflow steps in order
    steps = [
        ("Initialize", "Set up workflow state and requirements"),
        ("Quality Check", "Assess content quality metrics"),
        ("Subject Classification", "Classify assignment by subject"),
        ("Specialized Processing", "Apply subject-specific processing"),
        ("Grammar Analysis", "Analyze grammar and language quality"),
        ("Plagiarism Detection", "Detect potential plagiarism"),
        ("Relevance Analysis", "Analyze content relevance to source"),
        ("Content Grading", "Grade content on multiple criteria"),
        ("Summary Generation", "Generate comprehensive summary"),
        ("Quality Validation", "Validate processing quality"),
        ("Error Recovery", "Attempt to recover from errors"),
        ("Results Aggregation", "Aggregate all processing results"),
        ("Finalize", "Finalize workflow execution")
    ]

    for i, (step_name, description) in enumerate(steps, 1):
        print(f"{i:2d}. {step_name:<20} - {description}")

    print("\n🔄 Conditional Routing:")
    print("   • Each step determines the next step based on:")
    print("     - Processing requirements")
    print("     - Current state")
    print("     - Success/failure of previous steps")
    print("     - Error recovery needs")

    print(f"\n📊 Workflow Features:")
    print("   • State persistence with MemorySaver")
    print("   • Error recovery and retry logic")
    print("   • Quality validation")
    print("   • Subject-specific specialized processing")
    print("   • Multi-language support")
    print("   • Comprehensive result aggregation")


def get_current_graph_json() -> dict:
    """
    Return the current graph structure as JSON.
    Useful for programmatic access to the graph structure.
    """
    nodes, edges, graph_dict = get_workflow_graph_info()

    return {
        "nodes": nodes,
        "edges": edges,
        "workflow_metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entry_point": "initialize",
            "exit_point": "finalize",
            "supports_checkpointing": True,
            "supports_error_recovery": True
        },
        "raw_graph": graph_dict
    }


if __name__ == "__main__":
    # Example usage
    print("🚀 LangGraph Workflow Visualization")
    print("=" * 40)

    # Print text structure
    print_workflow_structure()

    print("\n" + "=" * 40)

    # Create visual graph
    summary = visualize_workflow_graph(
        save_path="workflow_graph.png",
        show_plot=False  # Set to True to display the plot
    )

    # Print JSON structure
    print("\n📄 JSON Structure:")
    graph_json = get_current_graph_json()
    print(json.dumps(graph_json["workflow_metadata"], indent=2))