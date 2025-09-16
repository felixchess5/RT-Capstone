#!/usr/bin/env python3
"""
Test script for specialized math and Spanish processors.
"""
import asyncio
import json
from assignment_orchestrator import create_assignment_orchestrator
from math_processor import create_math_processor
from spanish_processor import create_spanish_processor

# Sample math assignment
MATH_ASSIGNMENT = """
Name: John Smith
Date: 2024-09-16
Class: Algebra II
Subject: Mathematics

Assignment: Solve the following equations and show your work

1. Solve for x: 2x + 5 = 13
   First, I subtract 5 from both sides:
   2x + 5 - 5 = 13 - 5
   2x = 8
   Then divide by 2:
   x = 4

2. Find the derivative of f(x) = x² + 3x - 2
   Using the power rule:
   f'(x) = 2x + 3

3. Calculate: ∫(x² + 1)dx
   Using integration rules:
   ∫(x² + 1)dx = x³/3 + x + C

Step-by-step work is shown for each problem. Mathematical notation is used correctly.
"""

# Sample Spanish assignment
SPANISH_ASSIGNMENT = """
Nombre: María González
Fecha: 16 de septiembre, 2024
Clase: Español II
Materia: Español

Tarea: Escribe sobre la cultura española

La cultura española es muy rica y diversa. España tiene muchas tradiciones importantes como el flamenco, las tapas, y la siesta. En España se celebran muchas fiestas durante el año.

Las comidas españolas son deliciosas. Me gusta mucho la paella, el gazpacho, y los churros. Los españoles comen tarde en comparación con otros países.

El idioma español se habla en muchos países del mundo. Es importante aprender español porque hay muchas oportunidades de trabajo.

¿Qué tradiciones españolas te parecen más interesantes? A mí me fascina el flamenco porque es una expresión artística muy emocional.

En conclusión, la cultura española tiene mucho que ofrecer al mundo.
"""

# Sample general assignment
GENERAL_ASSIGNMENT = """
Name: Sarah Johnson
Date: September 16, 2024
Class: English Literature
Subject: English

Assignment: Analyze the themes in Romeo and Juliet

Romeo and Juliet by William Shakespeare explores several important themes. The most prominent theme is love, particularly young love and its intensity. The play shows how love can be both beautiful and destructive.

Another major theme is fate versus free will. The characters seem destined for tragedy, but their own choices also contribute to their downfall. The concept of feuding families represents how hatred can be passed down through generations.

The theme of death is woven throughout the play. Death is portrayed not just as an ending, but as a consequence of the characters' actions and the society they live in.

Shakespeare uses various literary devices to convey these themes, including foreshadowing, irony, and symbolism. The balcony scene, for example, symbolizes the separation between the lovers due to their families' feud.

In conclusion, Romeo and Juliet remains relevant today because its themes of love, conflict, and choice are universal human experiences.
"""

async def test_math_processor():
    """Test the math processor with a sample assignment."""
    print("🧮 Testing Math Processor")
    print("=" * 50)

    math_processor = create_math_processor()

    # Analyze the assignment
    analysis = math_processor.analyze_math_assignment(MATH_ASSIGNMENT)
    print(f"Math Analysis:")
    print(f"  Problem types: {analysis['problem_types']}")
    print(f"  Equations found: {len(analysis['equations_found'])}")
    print(f"  Step-by-step present: {analysis['step_by_step_present']}")
    print(f"  Completeness score: {analysis['completeness_score']:.2f}")

    # Grade the assignment
    grading = math_processor.grade_math_assignment(MATH_ASSIGNMENT)
    print(f"\nMath Grading:")
    print(f"  Mathematical accuracy: {grading['mathematical_accuracy']:.2f}/10")
    print(f"  Problem solving approach: {grading['problem_solving_approach']:.2f}/10")
    print(f"  Notation clarity: {grading['notation_clarity']:.2f}/10")
    print(f"  Step-by-step work: {grading['step_by_step_work']:.2f}/10")
    print(f"  Overall score: {grading['overall_score']:.2f}/10")
    print(f"\nFeedback:")
    for feedback in grading['feedback']:
        print(f"  - {feedback}")

async def test_spanish_processor():
    """Test the Spanish processor with a sample assignment."""
    print("\n🇪🇸 Testing Spanish Processor")
    print("=" * 50)

    spanish_processor = create_spanish_processor()

    # Analyze the assignment
    analysis = spanish_processor.analyze_spanish_assignment(SPANISH_ASSIGNMENT)
    print(f"Spanish Analysis:")
    print(f"  Assignment type: {analysis.assignment_type.value}")
    print(f"  Vocabulary level: {analysis.vocabulary_level}")
    print(f"  Grammar errors: {len(analysis.grammar_errors)}")
    print(f"  Cultural references: {len(analysis.cultural_references)}")
    print(f"  Fluency score: {analysis.fluency_score:.2f}/100")
    print(f"  Verb conjugations found: {len(analysis.verb_conjugations)}")

    # Grade the assignment
    grading = spanish_processor.grade_spanish_assignment(SPANISH_ASSIGNMENT)
    print(f"\nSpanish Grading:")
    print(f"  Grammar accuracy: {grading['grammar_accuracy']:.2f}/10")
    print(f"  Vocabulary usage: {grading['vocabulary_usage']:.2f}/10")
    print(f"  Fluency/communication: {grading['fluency_communication']:.2f}/10")
    print(f"  Cultural understanding: {grading['cultural_understanding']:.2f}/10")
    print(f"  Overall score: {grading['overall_score']:.2f}/10")
    print(f"\nFeedback:")
    for feedback in grading['feedback']:
        print(f"  - {feedback}")

async def test_orchestrator():
    """Test the assignment orchestrator with all sample assignments."""
    print("\n🎯 Testing Assignment Orchestrator")
    print("=" * 50)

    orchestrator = create_assignment_orchestrator()

    test_assignments = [
        ("Math Assignment", MATH_ASSIGNMENT, {"subject": "mathematics", "class": "Algebra II"}),
        ("Spanish Assignment", SPANISH_ASSIGNMENT, {"subject": "español", "class": "Spanish II"}),
        ("General Assignment", GENERAL_ASSIGNMENT, {"subject": "english", "class": "English Literature"})
    ]

    for name, assignment, metadata in test_assignments:
        print(f"\n--- {name} ---")

        # Classify the assignment
        classification = orchestrator.classify_assignment(assignment, metadata)
        print(f"Classification:")
        print(f"  Subject: {classification.subject.value}")
        print(f"  Complexity: {classification.complexity.value}")
        print(f"  Specific type: {classification.specific_type}")
        print(f"  Confidence: {classification.confidence:.2f}")
        print(f"  Processing approach: {classification.processing_approach}")

        # Process with intelligent orchestrator
        try:
            result = await orchestrator.process_assignment(assignment, None, metadata)
            print(f"Processing Results:")
            print(f"  Processor used: {result['classification']['subject']}")
            print(f"  Overall score: {result['overall_score']:.2f}")
            print(f"  Specialized feedback items: {len(result['specialized_feedback'])}")

            if result['specialized_feedback']:
                print("  Sample feedback:")
                for i, feedback in enumerate(result['specialized_feedback'][:2]):
                    print(f"    {i+1}. {feedback}")

        except Exception as e:
            print(f"  Processing failed: {str(e)}")

async def test_individual_tools():
    """Test individual tools from the processors."""
    print("\n🔧 Testing Individual Tools")
    print("=" * 50)

    # Test equation solving
    print("Equation Solving:")
    math_processor = create_math_processor()
    equations = ["2x + 5 = 13", "x^2 - 4 = 0", "sin(x) = 0.5"]

    for eq in equations:
        try:
            solution = math_processor.solve_equation(eq)
            print(f"  {eq} → {solution.solution} (confidence: {solution.confidence:.2f})")
        except Exception as e:
            print(f"  {eq} → Error: {str(e)}")

    # Test Spanish grammar checking
    print("\nSpanish Grammar Check:")
    spanish_processor = create_spanish_processor()
    spanish_texts = [
        "El gato negro está en la mesa.",
        "La niña alto come manzanas.",  # Error: should be "alta"
        "Me gusta mucho el paella."     # Error: should be "la paella"
    ]

    for text in spanish_texts:
        errors = spanish_processor.check_grammar(text)
        print(f"  '{text}' → {len(errors)} errors")
        for error in errors:
            print(f"    - {error['description']}")

async def main():
    """Run all tests."""
    print("🧪 Testing Specialized Assignment Processors")
    print("=" * 80)

    try:
        await test_math_processor()
        await test_spanish_processor()
        await test_orchestrator()
        await test_individual_tools()

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())