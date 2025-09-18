#!/usr/bin/env python3
"""
Test script for subject-specific output functionality.
"""
import asyncio
import json
import os
from subject_output_manager import create_subject_output_manager
from assignment_orchestrator import create_assignment_orchestrator

# Sample assignment results with different subjects
SAMPLE_ASSIGNMENTS = [
    # Math assignment result
    {
        "student_name": "John Smith",
        "date_of_submission": "2024-09-16",
        "class": "Algebra II",
        "subject": "Mathematics",
        "overall_score": 8.5,
        "letter_grade": "B",
        "grammar_errors": 2,
        "summary": "Good mathematical work with clear step-by-step solutions.",
        "assignment_classification": {
            "subject": "mathematics",
            "complexity": "high_school",
            "specific_type": "algebra",
            "confidence": 0.95
        },
        "specialized_processing": {
            "classification": {"subject": "mathematics"},
            "processing_results": {
                "grading": {
                    "mathematical_accuracy": 8.5,
                    "problem_solving_approach": 9.0,
                    "notation_clarity": 7.0,
                    "step_by_step_work": 8.0,
                    "overall_score": 8.5
                },
                "analysis": {
                    "equations_found": ["2x + 5 = 13", "f'(x) = 2x + 3"],
                    "problem_types": ["algebra", "calculus"],
                    "step_by_step_present": True
                }
            }
        },
        "specialized_grades": {
            "mathematical_accuracy": 8.5,
            "problem_solving_approach": 9.0
        }
    },
    # Spanish assignment result
    {
        "student_name": "María González",
        "date_of_submission": "2024-09-16",
        "class": "Spanish II",
        "subject": "Spanish",
        "overall_score": 7.8,
        "letter_grade": "B",
        "grammar_errors": 1,
        "summary": "Excellent use of cultural references and good vocabulary variety.",
        "assignment_classification": {
            "subject": "spanish",
            "complexity": "middle_school",
            "specific_type": "culture",
            "confidence": 0.92
        },
        "specialized_processing": {
            "classification": {"subject": "spanish"},
            "processing_results": {
                "grading": {
                    "grammar_accuracy": 9.0,
                    "vocabulary_usage": 8.0,
                    "fluency_communication": 7.5,
                    "cultural_understanding": 8.5,
                    "overall_score": 7.8
                },
                "analysis": {
                    "vocabulary_level": "intermediate",
                    "grammar_errors_count": 1,
                    "cultural_references_count": 5,
                    "assignment_type": "culture"
                }
            }
        }
    },
    # English assignment result
    {
        "student_name": "Sarah Johnson",
        "date_of_submission": "2024-09-16",
        "class": "English Literature",
        "subject": "English",
        "overall_score": 9.2,
        "letter_grade": "A",
        "grammar_errors": 0,
        "summary": "Excellent literary analysis with strong thesis and supporting evidence.",
        "assignment_classification": {
            "subject": "english",
            "complexity": "high_school",
            "specific_type": "analysis",
            "confidence": 0.88
        }
    },
    # Another Math assignment
    {
        "student_name": "Alex Chen",
        "date_of_submission": "2024-09-16",
        "class": "Calculus",
        "subject": "Mathematics",
        "overall_score": 6.5,
        "letter_grade": "D",
        "grammar_errors": 0,
        "summary": "Partial solutions shown, needs more work on integration techniques.",
        "assignment_classification": {
            "subject": "mathematics",
            "complexity": "college",
            "specific_type": "calculus",
            "confidence": 0.98
        },
        "specialized_processing": {
            "classification": {"subject": "mathematics"},
            "processing_results": {
                "grading": {
                    "mathematical_accuracy": 6.0,
                    "problem_solving_approach": 7.0,
                    "notation_clarity": 6.5,
                    "step_by_step_work": 6.0,
                    "overall_score": 6.5
                },
                "analysis": {
                    "equations_found": ["∫(x² + 1)dx", "dy/dx = 2x"],
                    "problem_types": ["calculus"],
                    "step_by_step_present": False
                }
            }
        }
    },
    # General assignment
    {
        "student_name": "Emma Wilson",
        "date_of_submission": "2024-09-16",
        "class": "History",
        "subject": "History",
        "overall_score": 8.0,
        "letter_grade": "B",
        "grammar_errors": 3,
        "summary": "Good historical analysis with proper chronological understanding.",
        "assignment_classification": {
            "subject": "history",
            "complexity": "high_school",
            "specific_type": "chronological",
            "confidence": 0.75
        }
    }
]

async def test_subject_output_manager():
    """Test the subject output manager functionality."""
    print("🧪 Testing Subject-Specific Output Manager")
    print("=" * 60)

    # Create output manager
    output_manager = create_subject_output_manager("./test_output")

    print(f"📊 Sample Data Summary:")
    print(f"   Total assignments: {len(SAMPLE_ASSIGNMENTS)}")

    # Count by subject
    subject_counts = {}
    for assignment in SAMPLE_ASSIGNMENTS:
        subject = output_manager.determine_subject(assignment)
        subject_counts[subject.value] = subject_counts.get(subject.value, 0) + 1

    for subject, count in subject_counts.items():
        print(f"   {subject.capitalize()}: {count} assignments")

    print(f"\n📂 Exporting Subject-Specific Files...")

    # Export all subjects
    export_results = output_manager.export_all_subjects(SAMPLE_ASSIGNMENTS)

    print(f"\n✅ Export Results:")
    total_files = 0
    for subject, files in export_results.items():
        print(f"   📚 {subject.upper()}:")
        for file_path in files:
            print(f"      - {file_path}")
            total_files += 1

    print(f"\n📋 Total files created: {total_files}")

    # Test individual subject exports
    print(f"\n🔧 Testing Individual Subject Exports...")

    # Test math export
    print("   Testing Math export...")
    from subject_output_manager import OutputSubject
    math_csv = output_manager.export_subject_csv(SAMPLE_ASSIGNMENTS, OutputSubject.MATHEMATICS)
    math_json = output_manager.export_subject_json(SAMPLE_ASSIGNMENTS, OutputSubject.MATHEMATICS)
    print(f"      Math CSV: {math_csv}")
    print(f"      Math JSON: {math_json}")

    # Test Spanish export
    print("   Testing Spanish export...")
    spanish_csv = output_manager.export_subject_csv(SAMPLE_ASSIGNMENTS, OutputSubject.SPANISH)
    spanish_json = output_manager.export_subject_json(SAMPLE_ASSIGNMENTS, OutputSubject.SPANISH)
    print(f"      Spanish CSV: {spanish_csv}")
    print(f"      Spanish JSON: {spanish_json}")

    return export_results

async def test_subject_classification():
    """Test subject classification for different assignment types."""
    print(f"\n🎯 Testing Subject Classification")
    print("=" * 60)

    orchestrator = create_assignment_orchestrator()
    output_manager = create_subject_output_manager()

    test_texts = [
        ("Math: Solve 2x + 5 = 13 and show your work step by step.", {"subject": "algebra"}),
        ("Escribe sobre la cultura española y sus tradiciones.", {"subject": "español"}),
        ("Analyze the themes in Romeo and Juliet by Shakespeare.", {"subject": "english"}),
        ("Explain the process of photosynthesis in plants.", {"subject": "biology"}),
        ("Discuss the causes of World War I.", {"subject": "history"})
    ]

    for text, metadata in test_texts:
        print(f"\nText: {text[:50]}...")

        # Classify with orchestrator
        classification = orchestrator.classify_assignment(text, metadata)
        print(f"   Classification: {classification.subject.value} ({classification.confidence:.2f})")

        # Determine output subject
        mock_assignment = {
            "assignment_classification": {
                "subject": classification.subject.value
            }
        }
        output_subject = output_manager.determine_subject(mock_assignment)
        print(f"   Output files: {output_subject.value}_assignments.csv/json")

async def test_file_content():
    """Test the content of generated files."""
    print(f"\n📄 Testing File Content")
    print("=" * 60)

    # Check if files were created and have content
    test_files = [
        "./test_output/math_assignments.csv",
        "./test_output/spanish_assignments.csv",
        "./test_output/english_assignments.csv",
        "./test_output/export_summary.txt"
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"   ✅ {os.path.basename(file_path)}: {file_size} bytes")

            # Show first few lines for CSV files
            if file_path.endswith('.csv'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:3]  # First 3 lines
                    print(f"      Preview: {len(lines)} lines")
                    for i, line in enumerate(lines):
                        print(f"        {i+1}: {line.strip()[:60]}...")

        else:
            print(f"   ❌ {os.path.basename(file_path)}: Not found")

async def main():
    """Run all tests."""
    print("🧪 Subject-Specific Output System Tests")
    print("=" * 80)

    try:
        # Ensure test output directory exists
        os.makedirs("./test_output", exist_ok=True)

        # Run tests
        await test_subject_output_manager()
        await test_subject_classification()
        await test_file_content()

        print(f"\n✅ All tests completed successfully!")
        print(f"📁 Check the './test_output' folder for generated files")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())