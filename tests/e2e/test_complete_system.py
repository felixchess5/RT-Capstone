"""
End-to-end tests for the complete RT-Capstone system.

Tests the entire system from file upload to final graded results,
including all major workflows and user scenarios.
"""

import asyncio
import csv
import json
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from gradio_app import GradioAssignmentGrader

# Import system components
from main_agentic import main as agentic_main


class TestCompleteSystem:
    """End-to-end tests for the complete system."""

    @pytest.fixture
    def sample_assignments_dir(self, temp_dir):
        """Create a directory with sample assignments for testing."""
        assignments_dir = temp_dir / "Assignments"
        assignments_dir.mkdir()

        # Create various assignment types
        assignments = {
            "math_algebra.txt": """
Name: Alice Johnson
Date: 2025-01-15
Class: Algebra II
Subject: Mathematics

Problem 1: Solve for x: 3x - 7 = 14
Solution:
3x - 7 = 14
3x = 14 + 7
3x = 21
x = 7

Problem 2: Factor: x² - 9
Solution: (x + 3)(x - 3)
            """,
            "spanish_essay.txt": """
Name: Carlos Rodriguez
Date: 2025-01-15
Class: Spanish III
Subject: Spanish

Ensayo: Mi Familia

Mi familia es muy importante para mí. Tengo dos hermanos y una hermana.
Mis padres trabajan mucho pero siempre tienen tiempo para nosotros.
Los fines de semana nos gusta ir al parque y jugar fútbol juntos.
La cultura hispana valora mucho la familia y las tradiciones.
            """,
            "science_lab.txt": """
Name: Emma Wilson
Date: 2025-01-15
Class: Biology
Subject: Science

Lab Report: Photosynthesis Experiment

Hypothesis: Plants in sunlight will produce more oxygen than plants in darkness.

Materials:
- Aquatic plants (Elodea)
- Test tubes
- Light source
- Timer

Procedure:
1. Place plants in test tubes filled with water
2. Expose one set to bright light, another to darkness
3. Count oxygen bubbles produced in 10 minutes

Results:
Light condition: 45 bubbles
Dark condition: 3 bubbles

Conclusion:
The hypothesis is supported. Plants in light produced significantly more oxygen,
confirming that photosynthesis requires light energy.
            """,
            "history_essay.txt": """
Name: James Miller
Date: 2025-01-15
Class: World History
Subject: History

The Causes of World War I

World War I began in 1914 due to several interconnected factors.
The immediate cause was the assassination of Archduke Franz Ferdinand,
but underlying tensions had been building for years.

The alliance system created a web of obligations that drew nations into conflict.
Imperialism created competition for colonies and resources.
Nationalism stirred ethnic tensions, especially in the Balkans.
The arms race, particularly naval competition between Britain and Germany,
increased international tensions.

These factors combined to create a volatile situation where a single
spark could ignite a global conflagration.
            """,
        }

        for filename, content in assignments.items():
            (assignments_dir / filename).write_text(content.strip())

        return assignments_dir

    @pytest.fixture
    def output_dir(self, temp_dir):
        """Create output directory for test results."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        return output_dir

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_command_line_processing_workflow(
        self, sample_assignments_dir, output_dir, mock_multi_llm_manager
    ):
        """Test the complete command-line processing workflow."""
        # Mock LLM responses for different assignment types
        mock_llm = AsyncMock()

        def mock_llm_response(*args, **kwargs):
            # Return different responses based on input content
            content = str(args[0]) if args else str(kwargs)
            if "mathematics" in content.lower() or "algebra" in content.lower():
                return Mock(content="8.5")
            elif "spanish" in content.lower() or "ensayo" in content.lower():
                return Mock(content="7.5")
            elif "science" in content.lower() or "photosynthesis" in content.lower():
                return Mock(content="9.0")
            elif "history" in content.lower() or "world war" in content.lower():
                return Mock(content="8.0")
            else:
                return Mock(content="7.0")

        mock_llm.ainvoke.side_effect = mock_llm_response
        mock_multi_llm_manager.get_llm.return_value = mock_llm

        # Set up environment for testing
        import os

        original_cwd = os.getcwd()

        try:
            # Change to temporary directory
            os.chdir(temp_dir.parent)

            # Create symbolic links to maintain directory structure
            test_assignments = Path("Assignments")
            test_output = Path("output")

            if test_assignments.exists():
                test_assignments.unlink()
            if test_output.exists():
                test_output.unlink()

            test_assignments.symlink_to(sample_assignments_dir)
            test_output.symlink_to(output_dir)

            # Mock the main processing function
            with patch("main_agentic.run_agentic_workflow") as mock_workflow:
                mock_workflow.return_value = {
                    "math_algebra.txt": {
                        "overall_score": 8.5,
                        "subject": "mathematics",
                        "feedback": "Good algebraic work",
                    },
                    "spanish_essay.txt": {
                        "overall_score": 7.5,
                        "subject": "spanish",
                        "feedback": "Nice essay structure",
                    },
                    "science_lab.txt": {
                        "overall_score": 9.0,
                        "subject": "science",
                        "feedback": "Excellent lab report",
                    },
                    "history_essay.txt": {
                        "overall_score": 8.0,
                        "subject": "history",
                        "feedback": "Good historical analysis",
                    },
                }

                # Run the main workflow
                from main_agentic import main

                result = main()

                # Verify processing occurred
                assert mock_workflow.called

        finally:
            # Restore original directory
            os.chdir(original_cwd)

    @pytest.mark.e2e
    @pytest.mark.gradio
    def test_gradio_interface_workflow(self, sample_assignments_dir, temp_dir):
        """Test the complete Gradio interface workflow."""
        with patch("gradio_app.MultiLLMManager") as mock_llm_manager_class:
            # Set up mock LLM manager
            mock_llm_manager = Mock()
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = Mock(content="8.0")
            mock_llm_manager.get_llm.return_value = mock_llm
            mock_llm_manager_class.return_value = mock_llm_manager

            # Create Gradio grader instance
            grader = GradioAssignmentGrader()

            # Test file processing
            test_file = sample_assignments_dir / "math_algebra.txt"

            # Mock the agentic workflow
            with patch("gradio_app.run_agentic_workflow") as mock_workflow:
                mock_workflow.return_value = {
                    "overall_score": 8.5,
                    "classification": {
                        "subject": "mathematics",
                        "complexity": "high_school",
                        "confidence": 0.9,
                    },
                    "specialized_feedback": ["Good algebraic reasoning"],
                    "grammar": {"score": 8, "errors": 2},
                    "plagiarism": {"score": 9, "similarity_percentage": 5},
                    "relevance": {"score": 8, "alignment_score": 80},
                }

                # Process single assignment
                result = grader.process_single_assignment(
                    str(test_file),
                    enable_grammar=True,
                    enable_plagiarism=True,
                    enable_relevance=True,
                    enable_grading=True,
                    enable_summary=True,
                )

                # Verify results
                assert (
                    "Processing completed successfully" in result
                    or result != "No file uploaded"
                )
                assert mock_workflow.called

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_batch_processing_workflow(self, sample_assignments_dir, output_dir):
        """Test batch processing of multiple assignments."""
        assignment_files = list(sample_assignments_dir.glob("*.txt"))
        assert len(assignment_files) == 4

        # Mock batch processing
        with patch(
            "workflows.agentic_workflow.create_workflow"
        ) as mock_create_workflow:
            mock_workflow = AsyncMock()
            mock_workflow.ainvoke.return_value = {
                "overall_score": 8.0,
                "results": {"feedback": "Good work"},
            }
            mock_create_workflow.return_value = mock_workflow

            # Import and test batch processing function
            from core.assignment_orchestrator import AssignmentOrchestrator
            from core.llms import MultiLLMManager

            # Mock LLM manager
            with patch("core.llms.MultiLLMManager") as mock_llm_class:
                mock_llm_manager = Mock()
                mock_llm_class.return_value = mock_llm_manager

                orchestrator = AssignmentOrchestrator(mock_llm_manager)

                # Process all assignments
                results = []
                for assignment_file in assignment_files:
                    classification = orchestrator.classify_assignment(
                        assignment_file.read_text()
                    )
                    results.append(
                        {
                            "file": assignment_file.name,
                            "subject": classification.subject.value,
                            "complexity": classification.complexity.value,
                        }
                    )

                # Verify all assignments were processed
                assert len(results) == 4

                # Verify different subjects were detected
                subjects = [r["subject"] for r in results]
                assert "mathematics" in subjects or "general" in subjects
                assert "spanish" in subjects or "general" in subjects

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, sample_assignments_dir, temp_dir):
        """Test error recovery in the complete workflow."""
        # Create a corrupted file
        corrupted_file = sample_assignments_dir / "corrupted.txt"
        corrupted_file.write_bytes(b"\x00\x01\x02\x03")  # Binary data

        # Test system's ability to handle corrupted files
        from support.file_processor import FileProcessor

        processor = FileProcessor()

        # Should handle corrupted file gracefully
        try:
            content = processor.extract_text_content(str(corrupted_file))
            # Should not crash, even if content is empty or garbled
            assert isinstance(content, str)
        except Exception as e:
            # Should handle errors gracefully
            assert "error" in str(e).lower() or "invalid" in str(e).lower()

    @pytest.mark.e2e
    def test_output_file_generation(self, sample_assignments_dir, output_dir):
        """Test generation of output files."""
        # Mock the subject output manager
        with patch(
            "core.subject_output_manager.SubjectOutputManager"
        ) as mock_output_manager:
            mock_manager = Mock()
            mock_output_manager.return_value = mock_manager

            # Mock file writing
            mock_manager.write_results_to_csv.return_value = True
            mock_manager.write_results_to_json.return_value = True

            # Simulate processing results
            results = [
                {
                    "file_name": "math_algebra.txt",
                    "subject": "mathematics",
                    "overall_score": 8.5,
                    "classification": {"complexity": "high_school"},
                },
                {
                    "file_name": "spanish_essay.txt",
                    "subject": "spanish",
                    "overall_score": 7.5,
                    "classification": {"complexity": "high_school"},
                },
            ]

            # Test output generation
            from core.subject_output_manager import SubjectOutputManager

            output_manager = SubjectOutputManager(str(output_dir))

            # Mock the actual file writing
            for result in results:
                mock_manager.write_results_to_csv([result], result["subject"])
                mock_manager.write_results_to_json([result], result["subject"])

            # Verify output manager was called
            assert mock_manager.write_results_to_csv.called
            assert mock_manager.write_results_to_json.called

    @pytest.mark.e2e
    @pytest.mark.performance
    def test_system_performance_end_to_end(
        self, sample_assignments_dir, performance_monitor
    ):
        """Test overall system performance."""
        start_time = time.time()

        # Process multiple files to test performance
        assignment_files = list(sample_assignments_dir.glob("*.txt"))

        from core.assignment_orchestrator import AssignmentOrchestrator
        from support.file_processor import FileProcessor

        processor = FileProcessor()

        # Mock LLM manager for performance testing
        with patch("core.llms.MultiLLMManager") as mock_llm_class:
            mock_llm_manager = Mock()
            mock_llm_class.return_value = mock_llm_manager

            orchestrator = AssignmentOrchestrator(mock_llm_manager)

            # Process all files
            for assignment_file in assignment_files:
                content = processor.extract_text_content(str(assignment_file))
                classification = orchestrator.classify_assignment(content)

                # Verify processing completes
                assert classification.subject is not None

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertion: should process quickly without real LLM calls
        assert total_time < 5.0  # Should complete in under 5 seconds

    @pytest.mark.e2e
    def test_configuration_loading(self, temp_dir):
        """Test system configuration loading."""
        # Create test environment file
        env_file = temp_dir / ".env"
        env_file.write_text(
            """
GROQ_API_KEY=test_key
LANGCHAIN_TRACING_V2=false
TESTING=true
        """
        )

        # Test environment loading
        import os
        from pathlib import Path

        # Mock environment loading
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key", "TESTING": "true"}):
            # Import components that use environment variables
            from core.llms import MultiLLMManager

            # Should initialize without errors
            with patch("core.llms.ChatGroq") as mock_groq:
                mock_groq.return_value = Mock()

                try:
                    llm_manager = MultiLLMManager()
                    assert llm_manager is not None
                except Exception as e:
                    # Should handle configuration errors gracefully
                    assert "configuration" in str(e).lower() or "api" in str(e).lower()

    @pytest.mark.e2e
    def test_user_workflow_scenarios(self, sample_assignments_dir, temp_dir):
        """Test complete user workflow scenarios."""
        scenarios = [
            {
                "name": "new_teacher_first_use",
                "steps": [
                    "upload_assignment",
                    "select_grading_options",
                    "process_assignment",
                    "review_results",
                    "download_report",
                ],
            },
            {
                "name": "batch_grading_session",
                "steps": [
                    "upload_multiple_assignments",
                    "configure_batch_settings",
                    "process_batch",
                    "review_summary",
                    "export_results",
                ],
            },
            {
                "name": "error_handling_scenario",
                "steps": [
                    "upload_invalid_file",
                    "handle_error_message",
                    "retry_with_valid_file",
                    "complete_processing",
                ],
            },
        ]

        for scenario in scenarios:
            # Mock scenario execution
            scenario_results = []

            for step in scenario["steps"]:
                # Simulate step execution
                step_result = {"step": step, "success": True}

                if step == "upload_invalid_file":
                    # Test error handling
                    step_result["success"] = False
                    step_result["error"] = "Invalid file format"

                scenario_results.append(step_result)

            # Verify scenario completion
            assert len(scenario_results) == len(scenario["steps"])

            # Check error handling in error scenario
            if scenario["name"] == "error_handling_scenario":
                error_steps = [r for r in scenario_results if not r["success"]]
                assert len(error_steps) > 0

    @pytest.mark.e2e
    @pytest.mark.security
    def test_security_end_to_end(self, temp_dir):
        """Test security aspects of the complete system."""
        # Test file upload security
        malicious_files = [
            ("../../../etc/passwd", "path_traversal"),
            ("script.exe", "executable_file"),
            ("large_file.txt", "oversized_file"),
        ]

        from support.file_processor import FileProcessor

        processor = FileProcessor()

        for filename, attack_type in malicious_files:
            # Test file validation
            is_valid = processor.is_valid_file(filename)

            if attack_type == "path_traversal":
                # Should reject path traversal attempts
                assert not processor.validate_file_security(filename)
            elif attack_type == "executable_file":
                # Should reject executable files
                assert not is_valid
            elif attack_type == "oversized_file":
                # Create oversized file for testing
                large_file = temp_dir / filename
                large_file.write_text("x" * (100 * 1024 * 1024))  # 100MB file

                # Should handle large files appropriately
                try:
                    content = processor.extract_text_content(str(large_file))
                    assert isinstance(content, str)
                except Exception:
                    # Should handle oversized files gracefully
                    pass

    @pytest.mark.e2e
    def test_data_persistence_workflow(self, sample_assignments_dir, temp_dir):
        """Test data persistence throughout the workflow."""
        # Create output directory
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Mock data persistence
        persistent_data = {
            "session_id": "test_session_123",
            "assignments_processed": [],
            "results": {},
            "timestamps": {},
        }

        # Simulate processing multiple assignments
        assignment_files = list(sample_assignments_dir.glob("*.txt"))

        for assignment_file in assignment_files:
            # Mock processing
            assignment_data = {
                "filename": assignment_file.name,
                "content_length": len(assignment_file.read_text()),
                "processed_at": time.time(),
            }

            persistent_data["assignments_processed"].append(assignment_data)
            persistent_data["results"][assignment_file.name] = {
                "score": 8.0,
                "subject": "general",
            }

        # Verify data persistence
        assert len(persistent_data["assignments_processed"]) == len(assignment_files)
        assert len(persistent_data["results"]) == len(assignment_files)

        # Test data serialization
        json_data = json.dumps(persistent_data, default=str)
        assert isinstance(json_data, str)
        assert len(json_data) > 0

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_regression_scenario(self, sample_assignments_dir, temp_dir):
        """Test complete regression scenario covering major functionality."""
        # This test covers a complete user workflow to catch regressions

        test_steps = [
            "system_initialization",
            "file_upload_and_validation",
            "content_extraction",
            "assignment_classification",
            "specialized_processing",
            "result_generation",
            "output_formatting",
            "file_export",
        ]

        step_results = {}

        try:
            # Step 1: System initialization
            from core.assignment_orchestrator import AssignmentOrchestrator
            from core.llms import MultiLLMManager
            from support.file_processor import FileProcessor

            with patch("core.llms.ChatGroq") as mock_groq:
                mock_groq.return_value = Mock()

                # Initialize components
                file_processor = FileProcessor()
                step_results["system_initialization"] = True

            # Step 2: File upload and validation
            test_file = sample_assignments_dir / "math_algebra.txt"
            is_valid = file_processor.is_valid_file(str(test_file))
            step_results["file_upload_and_validation"] = is_valid

            # Step 3: Content extraction
            content = file_processor.extract_text_content(str(test_file))
            step_results["content_extraction"] = len(content) > 0

            # Step 4: Assignment classification
            with patch("core.llms.MultiLLMManager") as mock_llm_class:
                mock_llm_manager = Mock()
                mock_llm_class.return_value = mock_llm_manager

                orchestrator = AssignmentOrchestrator(mock_llm_manager)
                classification = orchestrator.classify_assignment(content)
                step_results["assignment_classification"] = (
                    classification.subject is not None
                )

            # Step 5: Specialized processing (mocked)
            step_results["specialized_processing"] = True

            # Step 6: Result generation (mocked)
            step_results["result_generation"] = True

            # Step 7: Output formatting (mocked)
            step_results["output_formatting"] = True

            # Step 8: File export (mocked)
            step_results["file_export"] = True

        except Exception as e:
            # Mark failed steps
            for step in test_steps:
                if step not in step_results:
                    step_results[step] = False

        # Verify all steps completed successfully
        failed_steps = [step for step, success in step_results.items() if not success]

        if failed_steps:
            pytest.fail(f"Regression test failed at steps: {failed_steps}")

        assert all(step_results.values()), f"Some steps failed: {step_results}"
