"""
Simplified graph visualization for LangGraph workflow.
Shows the main workflow path without overwhelming conditional edges.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from workflows.agentic_workflow import build_agentic_workflow


def create_simplified_workflow_graph():
    """Create a simplified representation of the workflow showing main flow."""

    # Define the main workflow sequence
    main_workflow = [
        ("START", "initialize"),
        ("initialize", "quality_check"),
        ("quality_check", "subject_classification"),
        ("subject_classification", "specialized_processing"),
        ("specialized_processing", "grammar_analysis"),
        ("grammar_analysis", "plagiarism_detection"),
        ("plagiarism_detection", "relevance_analysis"),
        ("relevance_analysis", "content_grading"),
        ("content_grading", "summary_generation"),
        ("summary_generation", "quality_validation"),
        ("quality_validation", "results_aggregation"),
        ("results_aggregation", "finalize"),
        ("finalize", "END")
    ]

    # Add conditional paths (simplified)
    conditional_paths = [
        ("quality_validation", "error_recovery"),
        ("error_recovery", "results_aggregation"),
        ("subject_classification", "grammar_analysis"),  # Skip specialized processing
    ]

    return main_workflow, conditional_paths


def visualize_simple_workflow(save_path: str = "simple_workflow.png", show_plot: bool = False):
    """Create a clean, readable visualization of the workflow."""

    print("🎨 Creating simplified workflow visualization...")

    # Get simplified workflow
    main_flow, conditional_flow = create_simplified_workflow_graph()

    # Create NetworkX graph
    G = nx.DiGraph()

    # Add main workflow edges
    G.add_edges_from(main_flow)

    # Add conditional edges with different style
    for edge in conditional_flow:
        G.add_edge(edge[0], edge[1], style='conditional')

    # Create figure
    plt.figure(figsize=(20, 12))

    # Use hierarchical layout
    try:
        # Create a custom hierarchical layout
        pos = {}
        nodes_by_level = {
            0: ["START"],
            1: ["initialize"],
            2: ["quality_check"],
            3: ["subject_classification"],
            4: ["specialized_processing", "grammar_analysis"],
            5: ["plagiarism_detection"],
            6: ["relevance_analysis"],
            7: ["content_grading"],
            8: ["summary_generation"],
            9: ["quality_validation", "error_recovery"],
            10: ["results_aggregation"],
            11: ["finalize"],
            12: ["END"]
        }

        for level, nodes in nodes_by_level.items():
            for i, node in enumerate(nodes):
                if node in G.nodes():
                    x = i - (len(nodes) - 1) / 2  # Center nodes at each level
                    y = -level  # Top to bottom
                    pos[node] = (x, y)

    except Exception as e:
        print(f"Using fallback layout: {e}")
        pos = nx.spring_layout(G, k=3, iterations=100)

    # Define colors for different node types
    node_colors = {
        'START': '#4CAF50',
        'initialize': '#4CAF50',
        'quality_check': '#2196F3',
        'subject_classification': '#FF9800',
        'specialized_processing': '#9C27B0',
        'grammar_analysis': '#2196F3',
        'plagiarism_detection': '#F44336',
        'relevance_analysis': '#2196F3',
        'content_grading': '#4CAF50',
        'summary_generation': '#795548',
        'quality_validation': '#607D8B',
        'error_recovery': '#FF5722',
        'results_aggregation': '#3F51B5',
        'finalize': '#9E9E9E',
        'END': '#9E9E9E'
    }

    # Create labels
    labels = {}
    for node in G.nodes():
        if node in ['START', 'END']:
            labels[node] = node
        else:
            # Split long labels for better readability
            label = node.replace('_', '\n').title()
            if len(label) > 15:
                words = label.split('\n')
                if len(words) > 1:
                    labels[node] = '\n'.join(words[:2])  # Take first 2 words
                else:
                    labels[node] = label[:12] + '...' if len(label) > 12 else label
            else:
                labels[node] = label

    # Get node colors for each node
    node_list = list(G.nodes())
    colors = [node_colors.get(node, '#E0E0E0') for node in node_list]

    # Draw the complete graph using the simplified nx.draw function
    nx.draw(G, pos,
            node_color=colors,
            node_size=3000,
            with_labels=True,
            labels=labels,
            font_size=8,
            font_weight='bold',
            edge_color='black',
            width=1.5,
            alpha=0.9,
            arrows=True,
            arrowsize=20)

    # Add title
    plt.title('LangGraph Agentic Workflow - Simplified View',
              fontsize=18, fontweight='bold', pad=30)

    # Create legend
    legend_elements = [
        mpatches.Patch(color='#4CAF50', label='Start/End/Grading'),
        mpatches.Patch(color='#2196F3', label='Analysis'),
        mpatches.Patch(color='#FF9800', label='Classification'),
        mpatches.Patch(color='#9C27B0', label='Specialized Processing'),
        mpatches.Patch(color='#F44336', label='Plagiarism Detection'),
        mpatches.Patch(color='#795548', label='Summary Generation'),
        mpatches.Patch(color='#607D8B', label='Validation'),
        mpatches.Patch(color='#FF5722', label='Error Recovery'),
        mpatches.Patch(color='#3F51B5', label='Results Aggregation'),
        mpatches.Patch(color='#9E9E9E', label='Finalization')
    ]

    plt.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1, 0.5),
              fontsize=12)

    # Add flow type legend
    flow_legend = [
        plt.Line2D([0], [0], color='black', lw=2, label='Main Flow'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Conditional/Recovery')
    ]

    plt.legend(handles=flow_legend,
              loc='upper left',
              bbox_to_anchor=(0, 1),
              fontsize=12)

    # Remove axes
    plt.axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📁 Simplified graph saved to: {save_path}")
        except Exception as e:
            print(f"❌ Failed to save graph: {e}")

    # Show
    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"⚠️ Cannot display plot: {e}")

    plt.close()  # Clean up

    return f"""
🎯 Simplified Workflow Overview:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Main Flow: {len(main_flow)} steps
Conditional Paths: {len(conditional_flow)} alternatives

Key Features:
• Linear main processing flow
• Conditional routing for specialized processing
• Error recovery mechanisms
• Quality validation checkpoints

The workflow processes assignments through:
1. Initialization and quality assessment
2. Subject classification and specialized processing
3. Multi-faceted analysis (grammar, plagiarism, relevance)
4. Grading and summary generation
5. Quality validation and result aggregation
"""


if __name__ == "__main__":
    result = visualize_simple_workflow(
        save_path="simple_workflow_graph.png",
        show_plot=False
    )
    print(result)