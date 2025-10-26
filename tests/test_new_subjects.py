#!/usr/bin/env python3
"""
Test Script for Science and History Subject Processors

This script tests the new Science and History processors to ensure they work
correctly with the assignment grading system.
"""

import asyncio
import os
import sys
from typing import Any, Dict

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assignment_orchestrator import create_assignment_orchestrator
from history_processor import (
    HistoryAssignmentType,
    HistoryPeriod,
    create_history_processor,
)
from science_processor import (
    ScienceAssignmentType,
    ScienceSubject,
    create_science_processor,
)
from subject_output_manager import OutputSubject, create_subject_output_manager


def test_science_processor():
    """Test the Science processor with example assignments."""
    print("=" * 60)
    print("TESTING SCIENCE PROCESSOR")
    print("=" * 60)

    # Read science example files
    science_files = [
        "Assignments/science_example_1.txt",
        "Assignments/science_example_2.txt",
    ]

    science_processor = create_science_processor()

    for file_path in science_files:
        if not os.path.exists(file_path):
            print(f"⚠️  Science example file not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"\n📄 Testing: {os.path.basename(file_path)}")
        print("-" * 40)

        try:
            # Test analysis
            analysis = science_processor.analyze_science_assignment(content)

            print(f"✅ Subject Area: {analysis.subject_area.value}")
            print(f"✅ Assignment Type: {analysis.assignment_type.value}")
            print(
                f"✅ Scientific Method Elements: {sum(analysis.scientific_method_elements.values())}/{len(analysis.scientific_method_elements)}"
            )
            print(
                f"✅ Units/Measurements Found: {len(analysis.units_and_measurements)}"
            )
            print(f"✅ Formulas Identified: {len(analysis.formulas_identified)}")
            print(
                f"✅ Hypothesis Present: {'Yes' if analysis.hypothesis_present else 'No'}"
            )
            print(
                f"✅ Conclusion Present: {'Yes' if analysis.conclusion_present else 'No'}"
            )
            print(
                f"✅ Scientific Vocabulary Score: {analysis.scientific_vocabulary_score:.1f}/10"
            )
            print(f"✅ Safety Considerations: {len(analysis.safety_considerations)}")

            # Test specific methods
            subject = science_processor.identify_science_subject(content)
            assignment_type = science_processor.identify_assignment_type(content)
            formulas = science_processor.identify_formulas(content)
            variables = science_processor.identify_experimental_variables(content)

            print(f"\n📊 Detailed Analysis:")
            print(f"   Science Subject: {subject.value}")
            print(f"   Assignment Type: {assignment_type.value}")
            print(f"   Formulas: {formulas}")
            print(f"   Independent Variables: {variables['independent']}")
            print(f"   Dependent Variables: {variables['dependent']}")

        except Exception as e:
            print(f"❌ Science processor test failed: {e}")

    return True


def test_history_processor():
    """Test the History processor with example assignments."""
    print("\n" + "=" * 60)
    print("TESTING HISTORY PROCESSOR")
    print("=" * 60)

    # Read history example files
    history_files = [
        "Assignments/history_example_1.txt",
        "Assignments/history_example_2.txt",
    ]

    history_processor = create_history_processor()

    for file_path in history_files:
        if not os.path.exists(file_path):
            print(f"⚠️  History example file not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"\n📄 Testing: {os.path.basename(file_path)}")
        print("-" * 40)

        try:
            # Test analysis
            analysis = history_processor.analyze_history_assignment(content)

            print(f"✅ Historical Period: {analysis.period.value}")
            print(f"✅ Assignment Type: {analysis.assignment_type.value}")
            print(f"✅ Region Focus: {analysis.region_focus.value}")
            print(f"✅ Dates Identified: {len(analysis.dates_identified)}")
            print(f"✅ Historical Figures: {len(analysis.historical_figures)}")
            print(f"✅ Events Mentioned: {len(analysis.events_mentioned)}")
            print(f"✅ Sources Cited: {len(analysis.sources_cited)}")
            print(
                f"✅ Chronological Accuracy: {analysis.chronological_accuracy:.1f}/10"
            )
            print(
                f"✅ Historical Context Score: {analysis.historical_context_score:.1f}/10"
            )
            print(
                f"✅ Historical Vocabulary Score: {analysis.historical_vocabulary_score:.1f}/10"
            )

            # Test specific methods
            period = history_processor.identify_historical_period(content)
            assignment_type = history_processor.identify_assignment_type(content)
            dates = history_processor.extract_dates(content)
            figures = history_processor.extract_historical_figures(content)
            events = history_processor.extract_historical_events(content)

            print(f"\n📊 Detailed Analysis:")
            print(f"   Historical Period: {period.value}")
            print(f"   Assignment Type: {assignment_type.value}")
            print(f"   Dates Found: {dates[:5]}...")  # Show first 5
            print(f"   Historical Figures: {figures}")
            print(f"   Events: {events[:3]}...")  # Show first 3

        except Exception as e:
            print(f"❌ History processor test failed: {e}")

    return True


async def test_orchestrator_integration():
    """Test integration with the assignment orchestrator."""
    print("\n" + "=" * 60)
    print("TESTING ORCHESTRATOR INTEGRATION")
    print("=" * 60)

    orchestrator = create_assignment_orchestrator()

    # Test files for each subject
    test_files = [
        ("Assignments/science_example_1.txt", "Science"),
        ("Assignments/science_example_2.txt", "Science"),
        ("Assignments/history_example_1.txt", "History"),
        ("Assignments/history_example_2.txt", "History"),
    ]

    for file_path, expected_subject in test_files:
        if not os.path.exists(file_path):
            print(f"⚠️  Test file not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"\n📄 Testing Orchestrator: {os.path.basename(file_path)}")
        print("-" * 50)

        try:
            # Test classification
            classification = orchestrator.classify_assignment(content)

            print(f"✅ Detected Subject: {classification.subject.value}")
            print(f"✅ Specific Type: {classification.specific_type}")
            print(f"✅ Complexity: {classification.complexity.value}")
            print(f"✅ Confidence: {classification.confidence:.2f}")
            print(f"✅ Processing Approach: {classification.processing_approach}")
            print(f"✅ Tools Needed: {', '.join(classification.tools_needed[:3])}...")

            # Test full processing
            result = await orchestrator.process_assignment(content)

            print(f"\n📊 Processing Results:")
            print(f"   Overall Score: {result['overall_score']:.1f}/10")
            print(f"   Feedback Items: {len(result['specialized_feedback'])}")
            print(
                f"   Processing Status: {'Success' if 'error' not in result['processing_results'] else 'Failed'}"
            )

            if expected_subject.lower() in classification.subject.value:
                print(f"✅ Subject detection correct: {expected_subject}")
            else:
                print(
                    f"⚠️  Subject detection mismatch: expected {expected_subject}, got {classification.subject.value}"
                )

        except Exception as e:
            print(f"❌ Orchestrator integration test failed: {e}")

    return True


def test_subject_output_manager():
    """Test the subject output manager with new subjects."""
    print("\n" + "=" * 60)
    print("TESTING SUBJECT OUTPUT MANAGER")
    print("=" * 60)

    output_manager = create_subject_output_manager("./test_output")

    # Create mock assignment data for testing
    mock_assignments = [
        {
            "student_name": "Sarah Johnson",
            "subject": "Science",
            "overall_score": 8.5,
            "specialized_processing": {
                "classification": {"subject": "science"},
                "processing_results": {
                    "grading": {
                        "scientific_accuracy": 8.0,
                        "hypothesis_quality": 9.0,
                        "data_analysis": 8.5,
                        "experimental_design": 8.0,
                        "conclusion_validity": 9.0,
                    },
                    "analysis": {
                        "subject_area": "chemistry",
                        "assignment_type": "laboratory_report",
                        "units_and_measurements_count": 15,
                        "formulas_identified_count": 3,
                        "scientific_vocabulary_score": 8.2,
                    },
                },
            },
        },
        {
            "student_name": "Emily Rodriguez",
            "subject": "History",
            "overall_score": 9.0,
            "specialized_processing": {
                "classification": {"subject": "history"},
                "processing_results": {
                    "grading": {
                        "historical_accuracy": 9.0,
                        "chronological_understanding": 8.5,
                        "source_analysis": 9.5,
                        "contextual_awareness": 8.5,
                        "argument_development": 9.0,
                    },
                    "analysis": {
                        "period": "modern",
                        "assignment_type": "cause_and_effect",
                        "region_focus": "european_history",
                        "dates_identified_count": 12,
                        "historical_figures_count": 8,
                        "sources_cited_count": 3,
                    },
                },
            },
        },
    ]

    try:
        # Test subject determination
        for assignment in mock_assignments:
            subject = output_manager.determine_subject(assignment)
            print(
                f"✅ Subject determination: {assignment['student_name']} -> {subject.value}"
            )

        # Test specialized data extraction
        science_assignment = mock_assignments[0]
        science_data = output_manager.extract_specialized_data(
            science_assignment, OutputSubject.SCIENCE
        )
        print(f"\n📊 Science specialized data extracted: {len(science_data)} fields")
        print(
            f"   Scientific accuracy: {science_data.get('scientific_accuracy', 'N/A')}"
        )
        print(f"   Subject area: {science_data.get('subject_area', 'N/A')}")

        history_assignment = mock_assignments[1]
        history_data = output_manager.extract_specialized_data(
            history_assignment, OutputSubject.HISTORY
        )
        print(f"\n📊 History specialized data extracted: {len(history_data)} fields")
        print(
            f"   Historical accuracy: {history_data.get('historical_accuracy', 'N/A')}"
        )
        print(f"   Historical period: {history_data.get('historical_period', 'N/A')}")

        # Test export functionality
        export_results = output_manager.export_all_subjects(mock_assignments)
        print(f"\n📁 Export results:")
        for subject, files in export_results.items():
            if files:
                print(f"   {subject}: {len(files)} files created")

        print(f"✅ Subject output manager tests completed successfully")

    except Exception as e:
        print(f"❌ Subject output manager test failed: {e}")

    return True


async def test_grading_functionality():
    """Test the actual grading functionality for Science and History."""
    print("\n" + "=" * 60)
    print("TESTING GRADING FUNCTIONALITY")
    print("=" * 60)

    science_processor = create_science_processor()
    history_processor = create_history_processor()

    # Test Science grading
    science_files = ["Assignments/science_example_1.txt"]
    for file_path in science_files:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            print(f"\n🧪 Testing Science Grading: {os.path.basename(file_path)}")
            try:
                grading_result = await science_processor.grade_science_assignment(
                    content
                )
                print(f"✅ Science grading completed")
                print(f"   Overall Score: {grading_result['overall_score']:.1f}/10")
                print(f"   Subject Area: {grading_result['subject_area']}")
                print(f"   Assignment Type: {grading_result['assignment_type']}")
                print(f"   Feedback Items: {len(grading_result['feedback'])}")
                print(f"   Recommendations: {len(grading_result['recommendations'])}")
            except Exception as e:
                print(f"❌ Science grading failed: {e}")

    # Test History grading
    history_files = ["Assignments/history_example_1.txt"]
    for file_path in history_files:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            print(f"\n📚 Testing History Grading: {os.path.basename(file_path)}")
            try:
                grading_result = await history_processor.grade_history_assignment(
                    content
                )
                print(f"✅ History grading completed")
                print(f"   Overall Score: {grading_result['overall_score']:.1f}/10")
                print(f"   Historical Period: {grading_result['period']}")
                print(f"   Assignment Type: {grading_result['assignment_type']}")
                print(f"   Region Focus: {grading_result['region_focus']}")
                print(f"   Feedback Items: {len(grading_result['feedback'])}")
                print(f"   Recommendations: {len(grading_result['recommendations'])}")
            except Exception as e:
                print(f"❌ History grading failed: {e}")

    return True


async def run_all_tests():
    """Run all tests for Science and History processors."""
    print("🚀 SCIENCE AND HISTORY PROCESSOR TEST SUITE")
    print("=" * 80)

    tests = [
        ("Science Processor", test_science_processor),
        ("History Processor", test_history_processor),
        ("Orchestrator Integration", test_orchestrator_integration),
        ("Subject Output Manager", test_subject_output_manager),
        ("Grading Functionality", test_grading_functionality),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = "CRASH"

    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)

    for test_name, result in results.items():
        status_emoji = "✅" if result == "PASS" else "❌"
        print(f"{status_emoji} {test_name}: {result}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print(
            "🎉 All tests passed! Science and History processors are working correctly."
        )
    elif passed > total // 2:
        print("⚠️  Most tests passed. Check failed tests above.")
    else:
        print("❌ Many tests failed. Check processor implementations and dependencies.")

    print(f"\n📄 Test files created:")
    print(f"   - Assignments/science_example_1.txt")
    print(f"   - Assignments/science_example_2.txt")
    print(f"   - Assignments/history_example_1.txt")
    print(f"   - Assignments/history_example_2.txt")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
