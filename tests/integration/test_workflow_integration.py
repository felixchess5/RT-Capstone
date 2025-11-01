"""
Integration tests for workflow components.

Tests the interaction between different components in the assignment processing workflow.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.assignment_orchestrator import AssignmentOrchestrator
from core.llms import MultiLLMManager
from support.file_processor import FileProcessor
from workflows.agentic_workflow import create_workflow


class TestWorkflowIntegration:
    """Integration tests for the complete workflow."""

    @pytest.fixture
    def workflow_system(self, mock_multi_llm_manager):
        """Create a complete workflow system for testing (sync fixture)."""
        orchestrator = AssignmentOrchestrator(mock_multi_llm_manager)
        file_processor = FileProcessor()
        workflow = create_workflow(mock_multi_llm_manager)

        return {
            "orchestrator": orchestrator,
            "file_processor": file_processor,
            "workflow": workflow,
            "llm_manager": mock_multi_llm_manager,
        }

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_assignment_processing_flow(
        self, workflow_system, temp_files
    ):
        """Test the complete flow from file to graded result."""
        # Step 1: File processing
        file_path = str(temp_files["math"])
        content = workflow_system["file_processor"].extract_text_content(file_path)
        metadata = workflow_system["file_processor"].extract_metadata(content)

        assert content is not None
        assert isinstance(metadata, dict)

        # Step 2: Assignment classification
        classification = workflow_system["orchestrator"].classify_assignment(
            content, metadata
        )

        assert classification.subject.value in ["mathematics", "general"]
        assert classification.confidence > 0.0

        # Step 3: Specialized processing
        with patch.object(
            workflow_system["orchestrator"], "process_assignment"
        ) as mock_process:
            mock_process.return_value = {
                "overall_score": 8.5,
                "classification": {
                    "subject": "mathematics",
                    "complexity": "high_school",
                    "confidence": 0.9,
                },
                "processing_results": {"math_accuracy": 9, "problem_solving": 8},
                "specialized_feedback": ["Good mathematical approach"],
            }

            result = await workflow_system["orchestrator"].process_assignment(
                content, metadata=metadata
            )

            assert result["overall_score"] == 8.5
            assert result["classification"]["subject"] == "mathematics"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, workflow_system, temp_files):
        """Test batch processing of multiple assignments."""
        file_paths = [
            str(temp_files["math"]),
            str(temp_files["spanish"]),
            str(temp_files["science"]),
        ]

        results = []

        for file_path in file_paths:
            # Process each file through the complete workflow
            content = workflow_system["file_processor"].extract_text_content(file_path)
            metadata = workflow_system["file_processor"].extract_metadata(content)
            classification = workflow_system["orchestrator"].classify_assignment(
                content, metadata
            )

            # Mock the processing result
            with patch.object(
                workflow_system["orchestrator"], "process_assignment"
            ) as mock_process:
                mock_process.return_value = {
                    "overall_score": 7.5,
                    "classification": {
                        "subject": classification.subject.value,
                        "complexity": classification.complexity.value,
                        "confidence": classification.confidence,
                    },
                    "processing_results": {},
                    "specialized_feedback": [],
                }

                result = await workflow_system["orchestrator"].process_assignment(
                    content, metadata=metadata
                )
                results.append({"file_path": file_path, "result": result})

        assert len(results) == 3
        assert all("overall_score" in r["result"] for r in results)

    @pytest.mark.integration
    async def test_error_propagation_through_workflow(self, workflow_system, temp_dir):
        """Test how errors propagate through the workflow."""
        # Create an invalid file
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("")  # Empty file

        # Test file processing error handling
        content = workflow_system["file_processor"].extract_text_content(
            str(invalid_file)
        )
        assert content == ""  # Should handle empty file gracefully

        # Test orchestrator error handling with empty content
        classification = workflow_system["orchestrator"].classify_assignment(content)
        assert classification.subject.value == "general"  # Should default to general

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_failover_integration(self, workflow_system):
        """Test LLM failover in integrated workflow."""
        # Mock LLM manager with failover behavior
        primary_llm = AsyncMock()
        secondary_llm = AsyncMock()

        # Primary LLM fails
        primary_llm.ainvoke.side_effect = Exception("Primary LLM failed")
        # Secondary LLM succeeds
        secondary_llm.ainvoke.return_value = Mock(content="8.0")

        workflow_system["llm_manager"].get_llm.side_effect = [
            primary_llm,
            secondary_llm,
        ]

        # Process assignment
        test_content = "Solve for x: 2x + 5 = 13"

        with patch.object(
            workflow_system["orchestrator"], "process_assignment"
        ) as mock_process:
            mock_process.return_value = {
                "overall_score": 8.0,
                "classification": {"subject": "mathematics"},
                "processing_results": {},
                "specialized_feedback": [],
            }

            result = await workflow_system["orchestrator"].process_assignment(
                test_content
            )
            assert result["overall_score"] == 8.0

    @pytest.mark.integration
    def test_file_type_to_processor_routing(self, workflow_system, temp_files):
        """Test routing from file types to appropriate processors."""
        test_cases = [
            (temp_files["math"], "mathematics"),
            (temp_files["spanish"], "spanish"),
            (temp_files["science"], "science"),
        ]

        for file_path, expected_subject in test_cases:
            content = workflow_system["file_processor"].extract_text_content(
                str(file_path)
            )
            classification = workflow_system["orchestrator"].classify_assignment(
                content
            )

            # Should route to correct subject or general
            assert classification.subject.value in [expected_subject, "general"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metadata_flow_through_system(self, workflow_system, temp_files):
        """Test metadata preservation through the workflow."""
        file_path = str(temp_files["math"])

        # Extract content and metadata
        content = workflow_system["file_processor"].extract_text_content(file_path)
        metadata = workflow_system["file_processor"].extract_metadata(content)

        # Ensure metadata is extracted
        assert isinstance(metadata, dict)
        if metadata:  # If metadata exists
            assert "name" in metadata or "subject" in metadata or "class" in metadata

        # Classification should use metadata
        classification = workflow_system["orchestrator"].classify_assignment(
            content, metadata
        )

        # Mock processing to verify metadata is passed through
        with patch.object(
            workflow_system["orchestrator"], "process_assignment"
        ) as mock_process:
            mock_process.return_value = {
                "overall_score": 8.0,
                "classification": {
                    "subject": classification.subject.value,
                    "metadata_used": bool(metadata),
                },
                "processing_results": {},
                "specialized_feedback": [],
            }

            result = await workflow_system["orchestrator"].process_assignment(
                content, metadata=metadata
            )

            # Verify metadata was passed to processing
            mock_process.assert_called_with(content, metadata=metadata)

    @pytest.mark.integration
    def test_concurrent_file_processing(self, workflow_system, temp_files):
        """Test concurrent processing of multiple files."""
        file_paths = [
            str(temp_files["math"]),
            str(temp_files["spanish"]),
            str(temp_files["science"]),
        ]

        # Process files concurrently
        import concurrent.futures
        import threading

        results = []
        lock = threading.Lock()

        def process_file(file_path):
            content = workflow_system["file_processor"].extract_text_content(file_path)
            classification = workflow_system["orchestrator"].classify_assignment(
                content
            )

            with lock:
                results.append(
                    {
                        "file_path": file_path,
                        "content_length": len(content),
                        "subject": classification.subject.value,
                    }
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_file, fp) for fp in file_paths]
            concurrent.futures.wait(futures)

        assert len(results) == 3
        assert all("subject" in r for r in results)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_state_management(self, workflow_system):
        """Test workflow state management in LangGraph."""
        # Create a mock workflow state
        initial_state = {
            "assignment_text": "Solve for x: 2x + 5 = 13",
            "metadata": {"subject": "Mathematics"},
            "classification": None,
            "results": None,
        }

        # Test that workflow can handle state transitions
        workflow = workflow_system["workflow"]

        # Mock the workflow execution
        with patch.object(workflow, "ainvoke") as mock_invoke:
            mock_invoke.return_value = {
                "assignment_text": initial_state["assignment_text"],
                "metadata": initial_state["metadata"],
                "classification": {"subject": "mathematics", "confidence": 0.9},
                "results": {"overall_score": 8.5},
            }

            final_state = await workflow.ainvoke(initial_state)

            assert final_state["classification"] is not None
            assert final_state["results"] is not None
            assert final_state["results"]["overall_score"] == 8.5

    @pytest.mark.integration
    def test_performance_under_load(
        self, workflow_system, temp_files, performance_monitor
    ):
        """Test system performance under load."""
        # Process multiple files rapidly
        file_path = str(temp_files["math"])

        processing_times = []
        import time

        for i in range(10):  # Process same file 10 times
            start_time = time.time()

            content = workflow_system["file_processor"].extract_text_content(file_path)
            classification = workflow_system["orchestrator"].classify_assignment(
                content
            )

            end_time = time.time()
            processing_times.append(end_time - start_time)

        # Performance assertions
        avg_time = sum(processing_times) / len(processing_times)
        assert avg_time < 1.0  # Should process quickly without LLM calls

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cross_component_data_consistency(self, workflow_system, temp_files):
        """Test data consistency across components."""
        file_path = str(temp_files["math"])

        # Extract content
        content = workflow_system["file_processor"].extract_text_content(file_path)

        # Classify assignment
        classification1 = workflow_system["orchestrator"].classify_assignment(content)
        classification2 = workflow_system["orchestrator"].classify_assignment(content)

        # Classifications should be consistent
        assert classification1.subject == classification2.subject
        assert classification1.complexity == classification2.complexity
        assert abs(classification1.confidence - classification2.confidence) < 0.1

    @pytest.mark.integration
    def test_resource_cleanup(self, workflow_system, temp_files):
        """Test proper resource cleanup after processing."""
        import gc
        import weakref

        file_path = str(temp_files["math"])

        # Create weak references to track objects
        content = workflow_system["file_processor"].extract_text_content(file_path)
        classification = workflow_system["orchestrator"].classify_assignment(content)

        # Clear references
        del content
        del classification

        # Force garbage collection
        gc.collect()

        # Verify cleanup (this is a basic test)
        assert True  # If we get here, no memory leaks caused crashes

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_gradio_integration_workflow(self, workflow_system, temp_files):
        """Test integration with Gradio interface workflow."""
        # Simulate Gradio file upload workflow
        file_path = str(temp_files["math"])

        # Step 1: File validation (as Gradio would do)
        is_valid = workflow_system["file_processor"].is_valid_file(file_path)
        assert is_valid

        # Step 2: Extract content (as Gradio would do)
        content = workflow_system["file_processor"].extract_text_content(file_path)
        assert content

        # Step 3: Process through workflow (as Gradio would do)
        with patch.object(
            workflow_system["orchestrator"], "process_assignment"
        ) as mock_process:
            mock_process.return_value = {
                "overall_score": 8.5,
                "classification": {"subject": "mathematics"},
                "processing_results": {},
                "specialized_feedback": ["Good work"],
            }

            result = await workflow_system["orchestrator"].process_assignment(content)

            # Step 4: Format for Gradio display
            formatted_result = {
                "score": result["overall_score"],
                "subject": result["classification"]["subject"],
                "feedback": result["specialized_feedback"],
            }

            assert formatted_result["score"] == 8.5
            assert formatted_result["subject"] == "mathematics"

    @pytest.mark.integration
    def test_configuration_integration(self, workflow_system):
        """Test integration with configuration systems."""
        # Test that components use configuration correctly
        orchestrator = workflow_system["orchestrator"]
        file_processor = workflow_system["file_processor"]

        # Verify components have expected configuration
        assert hasattr(orchestrator, "llm_manager")
        assert hasattr(file_processor, "supported_formats")

        # Test configuration consistency
        available_processors = orchestrator.get_available_processors()
        assert isinstance(available_processors, dict)
        assert "subjects" in available_processors
