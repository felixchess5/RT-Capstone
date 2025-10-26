"""
Unit tests for MathProcessor class.

Tests mathematical analysis, equation solving, and math-specific grading.
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from processors.math_processor import MathProblemType, MathProcessor


class TestMathProcessor:
    """Test cases for MathProcessor."""

    def test_init(self, mock_multi_llm_manager):
        """Test math processor initialization."""
        processor = MathProcessor(mock_multi_llm_manager)

        assert processor.llm_manager == mock_multi_llm_manager
        assert hasattr(processor, "problem_patterns")
        assert hasattr(processor, "difficulty_indicators")

    @pytest.mark.parametrize(
        "text,expected_type",
        [
            ("Solve for x: 2x + 5 = 13", MathProblemType.ALGEBRA),
            ("Find the derivative of x²", MathProblemType.CALCULUS),
            ("Calculate the area of a triangle", MathProblemType.GEOMETRY),
            ("Find the mean of the dataset", MathProblemType.STATISTICS),
            ("Solve the differential equation", MathProblemType.CALCULUS),
            ("Factor the polynomial x² - 4", MathProblemType.ALGEBRA),
        ],
    )
    def test_classify_math_problem(self, math_processor, text, expected_type):
        """Test classification of different math problem types."""
        problem_type = math_processor.classify_math_problem(text)
        assert problem_type == expected_type

    def test_grade_math_assignment_basic(self, math_processor):
        """Test basic math assignment grading."""
        assignment_text = """
        Problem: Solve for x: 2x + 5 = 13
        Solution:
        2x + 5 = 13
        2x = 13 - 5
        2x = 8
        x = 4
        """

        with patch.object(math_processor.llm_manager, "get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = Mock(content="8.5")
            mock_get_llm.return_value = mock_llm

            result = math_processor.grade_math_assignment(assignment_text)

            assert isinstance(result, dict)
            assert "overall_score" in result
            assert "math_accuracy" in result
            assert "problem_solving_approach" in result
            assert "notation_clarity" in result
            assert "step_by_step_work" in result

    def test_extract_equations(self, math_processor):
        """Test equation extraction from text."""
        text = """
        Here are some equations:
        1. 2x + 3 = 7
        2. y = mx + b
        3. a² + b² = c²
        """

        equations = math_processor.extract_equations(text)

        assert isinstance(equations, list)
        assert len(equations) >= 3
        assert "2x + 3 = 7" in equations or "2x+3=7" in equations

    def test_validate_solution_steps(self, math_processor):
        """Test solution step validation."""
        correct_steps = ["2x + 5 = 13", "2x = 13 - 5", "2x = 8", "x = 4"]

        validation = math_processor.validate_solution_steps(correct_steps)

        assert isinstance(validation, dict)
        assert "is_valid" in validation
        assert "errors" in validation
        assert "clarity_score" in validation

    def test_check_mathematical_notation(self, math_processor):
        """Test mathematical notation checking."""
        text_with_good_notation = "x² + 2x - 3 = 0, where x ∈ ℝ"
        text_with_poor_notation = "x squared plus 2 times x minus 3 equals zero"

        good_score = math_processor.check_mathematical_notation(text_with_good_notation)
        poor_score = math_processor.check_mathematical_notation(text_with_poor_notation)

        assert isinstance(good_score, (int, float))
        assert isinstance(poor_score, (int, float))
        assert 0 <= good_score <= 10
        assert 0 <= poor_score <= 10
        assert good_score >= poor_score

    def test_analyze_problem_complexity(self, math_processor):
        """Test problem complexity analysis."""
        simple_problem = "What is 2 + 2?"
        complex_problem = (
            "Find the limit of (sin x - x)/x³ as x approaches 0 using L'Hôpital's rule"
        )

        simple_complexity = math_processor.analyze_problem_complexity(simple_problem)
        complex_complexity = math_processor.analyze_problem_complexity(complex_problem)

        assert isinstance(simple_complexity, dict)
        assert isinstance(complex_complexity, dict)
        assert "complexity_score" in simple_complexity
        assert "complexity_score" in complex_complexity
        assert (
            simple_complexity["complexity_score"]
            < complex_complexity["complexity_score"]
        )

    @pytest.mark.parametrize(
        "problem_text,expected_difficulty",
        [
            ("2 + 2 = ?", "elementary"),
            ("Solve for x: 2x + 5 = 13", "middle_school"),
            ("Find the derivative of x² + 3x", "high_school"),
            ("Evaluate the triple integral", "college"),
        ],
    )
    def test_assess_difficulty(self, math_processor, problem_text, expected_difficulty):
        """Test difficulty assessment for different problems."""
        difficulty = math_processor.assess_difficulty(problem_text)
        assert difficulty == expected_difficulty

    def test_symbolic_computation(self, math_processor):
        """Test symbolic computation capabilities."""
        try:
            import sympy

            equation = "x**2 - 4"
            factored = math_processor.symbolic_factor(equation)
            assert factored is not None
        except ImportError:
            pytest.skip("SymPy not available for symbolic computation")

    def test_evaluate_mathematical_reasoning(self, math_processor):
        """Test evaluation of mathematical reasoning."""
        reasoning_text = """
        To solve 2x + 5 = 13, I first subtract 5 from both sides
        to isolate the term with x. This gives me 2x = 8.
        Then I divide both sides by 2 to get x = 4.
        I can verify this by substituting back: 2(4) + 5 = 8 + 5 = 13 ✓
        """

        reasoning_score = math_processor.evaluate_mathematical_reasoning(reasoning_text)

        assert isinstance(reasoning_score, dict)
        assert "reasoning_quality" in reasoning_score
        assert "logical_flow" in reasoning_score
        assert "verification_present" in reasoning_score

    def test_detect_common_errors(self, math_processor):
        """Test detection of common mathematical errors."""
        error_examples = [
            "2x + 3x = 5x²",  # Incorrect like terms combination
            "√(a + b) = √a + √b",  # Incorrect radical distribution
            "(a + b)² = a² + b²",  # Missing cross term
        ]

        for error_text in error_examples:
            errors = math_processor.detect_common_errors(error_text)
            assert isinstance(errors, list)
            # Should detect at least one error in each example

    def test_grade_geometry_problem(self, math_processor):
        """Test grading of geometry-specific problems."""
        geometry_text = """
        Problem: Find the area of a triangle with base 6 and height 4.
        Solution: Area = (1/2) × base × height = (1/2) × 6 × 4 = 12 square units
        """

        with patch.object(math_processor.llm_manager, "get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = Mock(content="9.0")
            mock_get_llm.return_value = mock_llm

            result = math_processor.grade_math_assignment(geometry_text)

            assert result["problem_type"] == MathProblemType.GEOMETRY.value

    def test_grade_calculus_problem(self, math_processor):
        """Test grading of calculus-specific problems."""
        calculus_text = """
        Problem: Find the derivative of f(x) = x³ + 2x²
        Solution: f'(x) = 3x² + 4x
        """

        with patch.object(math_processor.llm_manager, "get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = Mock(content="8.5")
            mock_get_llm.return_value = mock_llm

            result = math_processor.grade_math_assignment(calculus_text)

            assert result["problem_type"] == MathProblemType.CALCULUS.value

    def test_error_handling(self, math_processor):
        """Test error handling in math processing."""
        # Test with empty input
        result = math_processor.grade_math_assignment("")
        assert isinstance(result, dict)
        assert "error" in result or "overall_score" in result

        # Test with invalid mathematical expressions
        invalid_text = "This is not a math problem at all!"
        result = math_processor.grade_math_assignment(invalid_text)
        assert isinstance(result, dict)

    def test_multiple_problems_in_assignment(self, math_processor):
        """Test handling assignments with multiple problems."""
        multi_problem_text = """
        Problem 1: Solve for x: 2x + 3 = 7
        Solution: x = 2

        Problem 2: Find the derivative of x²
        Solution: 2x

        Problem 3: Calculate 5! (factorial)
        Solution: 120
        """

        with patch.object(math_processor.llm_manager, "get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = Mock(content="8.0")
            mock_get_llm.return_value = mock_llm

            result = math_processor.grade_math_assignment(multi_problem_text)

            assert isinstance(result, dict)
            assert "overall_score" in result
            # Should handle multiple problems appropriately

    def test_mathematical_symbols_recognition(self, math_processor):
        """Test recognition of mathematical symbols."""
        symbol_text = "∫₀¹ x² dx = [x³/3]₀¹ = 1/3"

        symbols_found = math_processor.extract_mathematical_symbols(symbol_text)

        assert isinstance(symbols_found, list)
        assert len(symbols_found) > 0

    def test_unit_consistency_check(self, math_processor):
        """Test checking for unit consistency in problems."""
        consistent_units = "Distance = 5 meters, Time = 2 seconds, Speed = 2.5 m/s"
        inconsistent_units = "Distance = 5 meters, Time = 2 hours, Speed = 2.5 m/s"

        consistent_result = math_processor.check_unit_consistency(consistent_units)
        inconsistent_result = math_processor.check_unit_consistency(inconsistent_units)

        assert isinstance(consistent_result, dict)
        assert isinstance(inconsistent_result, dict)

    @pytest.mark.parametrize(
        "expression,expected_evaluation",
        [
            ("2 + 3", 5),
            ("10 - 4", 6),
            ("3 * 4", 12),
            ("15 / 3", 5),
        ],
    )
    def test_basic_arithmetic_evaluation(
        self, math_processor, expression, expected_evaluation
    ):
        """Test basic arithmetic evaluation."""
        result = math_processor.evaluate_expression(expression)
        assert result == expected_evaluation

    def test_performance_with_large_text(self, math_processor, performance_monitor):
        """Test performance with large mathematical text."""
        large_text = (
            """
        Problem: Solve the system of equations:
        2x + 3y = 7
        4x - y = 1
        """
            * 100
        )  # Repeat to create large text

        with patch.object(math_processor.llm_manager, "get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = Mock(content="7.5")
            mock_get_llm.return_value = mock_llm

            result = math_processor.grade_math_assignment(large_text)

            assert isinstance(result, dict)
            # Performance monitor will check if it takes too long

    def test_mathematical_proof_evaluation(self, math_processor):
        """Test evaluation of mathematical proofs."""
        proof_text = """
        Theorem: The sum of two even numbers is even.
        Proof: Let a and b be even numbers.
        Then a = 2m and b = 2n for some integers m and n.
        a + b = 2m + 2n = 2(m + n)
        Since (m + n) is an integer, a + b is even. QED.
        """

        proof_evaluation = math_processor.evaluate_mathematical_proof(proof_text)

        assert isinstance(proof_evaluation, dict)
        assert "proof_validity" in proof_evaluation
        assert "logical_structure" in proof_evaluation
