"""
Unit tests for AssignmentOrchestrator class.

Tests the core assignment classification and orchestration functionality.
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.assignment_orchestrator import (
    AssignmentClassification,
    AssignmentComplexity,
    AssignmentOrchestrator,
    SubjectType,
)


class TestAssignmentOrchestrator:
    """Test cases for AssignmentOrchestrator."""

    def test_init(self, mock_multi_llm_manager):
        """Test orchestrator initialization."""
        orchestrator = AssignmentOrchestrator(mock_multi_llm_manager)

        assert orchestrator.llm_manager == mock_multi_llm_manager
        assert hasattr(orchestrator, "math_processor")
        assert hasattr(orchestrator, "spanish_processor")
        assert hasattr(orchestrator, "science_processor")
        assert hasattr(orchestrator, "history_processor")

    @pytest.mark.parametrize(
        "text,expected_subject",
        [
            ("Solve for x: 2x + 5 = 13", SubjectType.MATHEMATICS),
            ("Ensayo sobre la cultura hispana", SubjectType.SPANISH),
            ("Lab report on photosynthesis", SubjectType.SCIENCE),
            ("World War II timeline analysis", SubjectType.HISTORY),
            ("Essay about literature", SubjectType.ENGLISH),
            ("Random text without clear subject", SubjectType.GENERAL),
        ],
    )
    def test_classify_assignment_subject_detection(
        self, assignment_orchestrator, text, expected_subject
    ):
        """Test subject detection in assignment classification."""
        classification = assignment_orchestrator.classify_assignment(text)

        assert isinstance(classification, AssignmentClassification)
        assert classification.subject == expected_subject
        assert 0.0 <= classification.confidence <= 1.0

    @pytest.mark.parametrize(
        "text,metadata,expected_complexity",
        [
            (
                "derivative of x^2",
                {"class": "Calculus AP"},
                AssignmentComplexity.HIGH_SCHOOL,
            ),
            ("2 + 2 = ?", {"class": "Grade 2"}, AssignmentComplexity.ELEMENTARY),
            (
                "integral calculus problem",
                {"class": "University Math"},
                AssignmentComplexity.COLLEGE,
            ),
            (
                "solve for x: x + 1 = 2",
                {"class": "Middle School Algebra"},
                AssignmentComplexity.MIDDLE_SCHOOL,
            ),
        ],
    )
    def test_complexity_classification(
        self, assignment_orchestrator, text, metadata, expected_complexity
    ):
        """Test complexity classification with different inputs."""
        classification = assignment_orchestrator.classify_assignment(text, metadata)

        assert classification.complexity == expected_complexity

    def test_classify_assignment_with_metadata(self, assignment_orchestrator):
        """Test classification with rich metadata."""
        text = "Solve quadratic equation: x² - 5x + 6 = 0"
        metadata = {
            "name": "John Doe",
            "date": "2025-01-15",
            "class": "Algebra II",
            "subject": "Mathematics",
        }

        classification = assignment_orchestrator.classify_assignment(text, metadata)

        assert classification.subject == SubjectType.MATHEMATICS
        assert classification.complexity in [
            AssignmentComplexity.HIGH_SCHOOL,
            AssignmentComplexity.COLLEGE,
        ]
        assert classification.confidence > 0.5

    def test_classify_assignment_without_metadata(self, assignment_orchestrator):
        """Test classification without metadata."""
        text = "Calculate the area of a circle with radius 5"

        classification = assignment_orchestrator.classify_assignment(text)

        assert classification.subject == SubjectType.MATHEMATICS
        assert isinstance(classification.complexity, AssignmentComplexity)

    def test_determine_subject_patterns(self, assignment_orchestrator):
        """Test subject determination using pattern matching."""
        # Test math patterns
        math_texts = [
            "solve equation 2x + 3 = 7",
            "calculate derivative of sin(x)",
            "find integral of x²",
            "prove theorem using algebra",
        ]

        for text in math_texts:
            classification = assignment_orchestrator.classify_assignment(text)
            assert classification.subject == SubjectType.MATHEMATICS

        # Test Spanish patterns
        spanish_texts = [
            "escriba un ensayo sobre la familia",
            "conjugar verbos regulares",
            "la cultura de México es rica",
        ]

        for text in spanish_texts:
            classification = assignment_orchestrator.classify_assignment(text)
            assert classification.subject == SubjectType.SPANISH

    def test_complexity_determination_advanced(self, assignment_orchestrator):
        """Test advanced complexity determination algorithm."""
        # College-level indicators
        college_text = (
            "Evaluate the limit using L'Hôpital's rule and multivariable calculus"
        )
        college_metadata = {"class": "Advanced Calculus"}

        classification = assignment_orchestrator.classify_assignment(
            college_text, college_metadata
        )
        assert classification.complexity == AssignmentComplexity.COLLEGE

        # Elementary-level indicators
        elementary_text = "Count the apples: 2 + 3 = ?"
        elementary_metadata = {"class": "Grade 1 Math"}

        classification = assignment_orchestrator.classify_assignment(
            elementary_text, elementary_metadata
        )
        assert classification.complexity == AssignmentComplexity.ELEMENTARY

    @pytest.mark.asyncio
    async def test_process_assignment_math(
        self, assignment_orchestrator, sample_math_assignment
    ):
        """Test processing a math assignment."""
        with patch.object(
            assignment_orchestrator.math_processor, "grade_math_assignment"
        ) as mock_grade:
            mock_grade.return_value = {
                "overall_score": 8.5,
                "feedback": ["Good solution approach"],
                "math_accuracy": 9,
                "problem_solving": 8,
            }

            result = await assignment_orchestrator.process_assignment(
                sample_math_assignment["text"],
                metadata=sample_math_assignment["metadata"],
            )

            assert result["overall_score"] == 8.5
            assert "classification" in result
            assert result["classification"]["subject"] == "mathematics"
            mock_grade.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_assignment_spanish(
        self, assignment_orchestrator, sample_spanish_assignment
    ):
        """Test processing a Spanish assignment."""
        with patch.object(
            assignment_orchestrator.spanish_processor, "grade_spanish_assignment"
        ) as mock_grade:
            mock_grade.return_value = {
                "overall_score": 7.5,
                "feedback": ["Good vocabulary usage"],
                "grammar_score": 8,
                "cultural_understanding": 7,
            }

            result = await assignment_orchestrator.process_assignment(
                sample_spanish_assignment["text"],
                metadata=sample_spanish_assignment["metadata"],
            )

            assert result["overall_score"] == 7.5
            assert result["classification"]["subject"] == "spanish"
            mock_grade.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_assignment_error_handling(self, assignment_orchestrator):
        """Test error handling during assignment processing."""
        with patch.object(
            assignment_orchestrator.math_processor, "grade_math_assignment"
        ) as mock_grade:
            mock_grade.side_effect = Exception("Processing error")

            result = await assignment_orchestrator.process_assignment(
                "Solve for x: 2x + 5 = 13", metadata={"subject": "Mathematics"}
            )

            assert "error" in result["processing_results"]
            assert result["overall_score"] == 5.0  # Default fallback score
            assert (
                "Unable to complete specialized analysis."
                in result["specialized_feedback"]
            )

    def test_generate_next_steps_math(self, assignment_orchestrator):
        """Test next steps generation for math assignments."""
        classification = AssignmentClassification(
            subject=SubjectType.MATHEMATICS,
            complexity=AssignmentComplexity.HIGH_SCHOOL,
            specific_type="algebra",
            confidence=0.9,
            language="en",
            tools_needed=["calculator"],
            processing_approach="step_by_step",
        )

        steps = assignment_orchestrator._generate_next_steps(classification, 6.5)

        assert isinstance(steps, list)
        assert any("algebra" in step.lower() for step in steps)

    def test_generate_next_steps_low_score(self, assignment_orchestrator):
        """Test next steps generation for low scores."""
        classification = AssignmentClassification(
            subject=SubjectType.MATHEMATICS,
            complexity=AssignmentComplexity.HIGH_SCHOOL,
            specific_type="algebra",
            confidence=0.9,
            language="en",
            tools_needed=[],
            processing_approach="standard",
        )

        steps = assignment_orchestrator._generate_next_steps(classification, 4.5)

        assert isinstance(steps, list)
        assert any("fundamental concepts" in step.lower() for step in steps)

    def test_get_available_processors(self, assignment_orchestrator):
        """Test getting available processor information."""
        processors = assignment_orchestrator.get_available_processors()

        assert isinstance(processors, dict)
        assert "subjects" in processors
        assert "math_problem_types" in processors
        assert "spanish_assignment_types" in processors
        assert "complexity_levels" in processors

        # Verify subject types are included
        subjects = processors["subjects"]
        assert "mathematics" in subjects
        assert "spanish" in subjects
        assert "science" in subjects
        assert "history" in subjects

    def test_routing_logic(self, assignment_orchestrator):
        """Test assignment routing to appropriate processors."""
        # Test math routing
        math_classification = assignment_orchestrator.classify_assignment(
            "Solve for x: 2x = 10"
        )
        assert math_classification.subject == SubjectType.MATHEMATICS

        # Test Spanish routing
        spanish_classification = assignment_orchestrator.classify_assignment(
            "Hola, ¿cómo estás?"
        )
        assert spanish_classification.subject == SubjectType.SPANISH

        # Test science routing
        science_classification = assignment_orchestrator.classify_assignment(
            "Hypothesis: Plants need sunlight"
        )
        assert science_classification.subject == SubjectType.SCIENCE

    @pytest.mark.parametrize("confidence_level", [0.1, 0.5, 0.9])
    def test_confidence_levels(self, assignment_orchestrator, confidence_level):
        """Test different confidence levels in classification."""
        # Mock the confidence calculation
        with patch.object(
            assignment_orchestrator, "_calculate_confidence"
        ) as mock_confidence:
            mock_confidence.return_value = confidence_level

            classification = assignment_orchestrator.classify_assignment("Sample text")
            assert classification.confidence == confidence_level

    def test_edge_cases(self, assignment_orchestrator):
        """Test edge cases and boundary conditions."""
        # Empty text
        classification = assignment_orchestrator.classify_assignment("")
        assert classification.subject == SubjectType.GENERAL

        # Very long text
        long_text = "word " * 10000
        classification = assignment_orchestrator.classify_assignment(long_text)
        assert isinstance(classification, AssignmentClassification)

        # Special characters and encoding
        special_text = "Résolvez l'équation: x² + 2x - 3 = 0 🧮"
        classification = assignment_orchestrator.classify_assignment(special_text)
        assert isinstance(classification, AssignmentClassification)

    def test_metadata_extraction(self, assignment_orchestrator):
        """Test metadata extraction and usage."""
        metadata = {
            "name": "Test Student",
            "date": "2025-01-15",
            "class": "Advanced Physics",
            "subject": "Science",
            "teacher": "Dr. Smith",
        }

        classification = assignment_orchestrator.classify_assignment(
            "Calculate the velocity of an object", metadata
        )

        assert classification.subject == SubjectType.SCIENCE
        # Metadata should influence classification
