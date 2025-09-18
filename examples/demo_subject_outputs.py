#!/usr/bin/env python3
"""
Demonstration of subject-specific output functionality.
Shows how assignments are automatically classified and exported to appropriate files.
"""
import asyncio
from subject_output_manager import create_subject_output_manager
from assignment_orchestrator import create_assignment_orchestrator

# Simple demo assignments
DEMO_ASSIGNMENTS = [
    {
        "student_name": "Alice",
        "subject": "Mathematics",
        "assignment_classification": {"subject": "mathematics"},
        "overall_score": 8.5,
        "specialized_processing": {
            "processing_results": {
                "grading": {"mathematical_accuracy": 8.5, "problem_solving_approach": 9.0}
            }
        }
    },
    {
        "student_name": "Bob",
        "subject": "Spanish",
        "assignment_classification": {"subject": "spanish"},
        "overall_score": 7.8,
        "specialized_processing": {
            "processing_results": {
                "grading": {"grammar_accuracy": 9.0, "vocabulary_usage": 8.0}
            }
        }
    },
    {
        "student_name": "Carol",
        "subject": "English",
        "assignment_classification": {"subject": "english"},
        "overall_score": 9.2
    }
]

async def demo():
    """Quick demonstration of subject-specific outputs."""
    print("📚 Subject-Specific Output Demo")
    print("=" * 40)

    # Create output manager
    output_manager = create_subject_output_manager("./demo_output")

    # Export to subject-specific files
    results = output_manager.export_all_subjects(DEMO_ASSIGNMENTS)

    print("✅ Files created:")
    for subject, files in results.items():
        print(f"   {subject}: {len(files)} files")

    print("\n📁 Check './demo_output' folder for:")
    print("   - math_assignments.csv")
    print("   - spanish_assignments.csv")
    print("   - english_assignments.csv")
    print("   - export_summary.txt")

if __name__ == "__main__":
    asyncio.run(demo())