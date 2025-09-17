"""
Assignment Orchestrator
Intelligent system to determine assignment types and route to appropriate specialized processors.
"""
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass
import asyncio

from math_processor import MathProcessor, MathProblemType, create_math_processor
from spanish_processor import SpanishProcessor, SpanishAssignmentType, create_spanish_processor
from science_processor import ScienceProcessor, ScienceAssignmentType, create_science_processor
from history_processor import HistoryProcessor, HistoryAssignmentType, create_history_processor
from language_support import detect_text_language

class SubjectType(Enum):
    """Main subject categories."""
    MATHEMATICS = "mathematics"
    SPANISH = "spanish"
    ENGLISH = "english"
    SCIENCE = "science"
    HISTORY = "history"
    GENERAL = "general"
    UNKNOWN = "unknown"

class AssignmentComplexity(Enum):
    """Assignment complexity levels."""
    ELEMENTARY = "elementary"
    MIDDLE_SCHOOL = "middle_school"
    HIGH_SCHOOL = "high_school"
    COLLEGE = "college"
    UNKNOWN = "unknown"

@dataclass
class AssignmentClassification:
    """Result of assignment classification."""
    subject: SubjectType
    complexity: AssignmentComplexity
    specific_type: str
    confidence: float
    language: str
    tools_needed: List[str]
    processing_approach: str

class AssignmentOrchestrator:
    """Intelligent orchestrator for routing assignments to specialized processors."""

    def __init__(self):
        self.math_processor = create_math_processor()
        self.spanish_processor = create_spanish_processor()
        self.science_processor = create_science_processor()
        self.history_processor = create_history_processor()

        # Subject identification patterns
        self.subject_patterns = {
            SubjectType.MATHEMATICS: [
                r'\b(?:solve|equation|calculate|formula|algebra|geometry|calculus|derivative|integral|theorem|proof)\b',
                r'\b(?:math|mathematics|arithmetic|trigonometry|statistics|probability)\b',
                r'[xyz]\s*[=+\-*/]\s*\d+',
                r'\b(?:graph|plot|function|variable|coefficient)\b',
                r'∫|∑|π|θ|α|β|γ|√|±|≤|≥|≠',
                r'\b(?:sin|cos|tan|log|ln|exp)\b',
                r'\$.*[xyz+\-*/=].*\$'
            ],
            SubjectType.SPANISH: [
                r'\b(?:español|spanish|castellano)\b',
                r'\b(?:conjuga|conjugate|verbo|verb|artículo|article)\b',
                r'\b(?:traduce|translate|significa|means)\b',
                r'\b(?:gramática|grammar|vocabulario|vocabulary)\b',
                r'¿[^?]*\?',  # Spanish question marks
                r'\b(?:cultura|culture|tradición|tradition)\b',
                r'[áéíóúñü]'  # Spanish accents
            ],
            SubjectType.ENGLISH: [
                r'\b(?:essay|composition|literature|grammar|writing)\b',
                r'\b(?:analyze|analysis|theme|character|plot)\b',
                r'\b(?:metaphor|simile|alliteration|symbolism)\b',
                r'\b(?:shakespeare|poetry|novel|short story)\b'
            ],
            SubjectType.SCIENCE: [
                r'\b(?:biology|chemistry|physics|science|experiment)\b',
                r'\b(?:molecule|atom|cell|organism|reaction)\b',
                r'\b(?:hypothesis|theory|data|observation)\b',
                r'\b(?:force|energy|motion|gravity|acceleration)\b'
            ],
            SubjectType.HISTORY: [
                r'\b(?:history|historical|century|era|period)\b',
                r'\b(?:war|battle|revolution|empire|civilization)\b',
                r'\b(?:ancient|medieval|renaissance|modern)\b',
                r'\d{3,4}\s*(?:AD|BC|CE|BCE)'
            ]
        }

        # Complexity indicators
        self.complexity_patterns = {
            AssignmentComplexity.ELEMENTARY: [
                r'\b(?:add|subtract|multiply|divide|basic|simple)\b',
                r'\b(?:color|count|name|identify)\b',
                r'\b(?:grade\s*[1-5]|elementary|primary)\b'
            ],
            AssignmentComplexity.MIDDLE_SCHOOL: [
                r'\b(?:fraction|decimal|percentage|ratio)\b',
                r'\b(?:grade\s*[6-8]|middle\s*school)\b',
                r'\b(?:explain|describe|compare)\b'
            ],
            AssignmentComplexity.HIGH_SCHOOL: [
                r'\b(?:algebra|geometry|trigonometry)\b',
                r'\b(?:grade\s*(?:9|10|11|12)|high\s*school|secondary)\b',
                r'\b(?:analyze|evaluate|synthesize)\b',
                r'\b(?:AP|advanced\s*placement)\b'
            ],
            AssignmentComplexity.COLLEGE: [
                r'\b(?:calculus|differential|linear\s*algebra)\b',
                r'\b(?:university|college|undergraduate|graduate)\b',
                r'\b(?:research|thesis|dissertation)\b',
                r'\b(?:critical\s*analysis|theoretical)\b'
            ]
        }

        # Tools mapping
        self.subject_tools = {
            SubjectType.MATHEMATICS: [
                'equation_solver', 'graph_plotter', 'symbolic_math',
                'calculus_solver', 'statistics_analyzer'
            ],
            SubjectType.SPANISH: [
                'grammar_checker', 'vocabulary_analyzer', 'conjugation_checker',
                'cultural_reference_detector', 'fluency_assessor'
            ],
            SubjectType.ENGLISH: [
                'grammar_checker', 'plagiarism_detector', 'writing_analyzer',
                'literature_analyzer', 'style_checker'
            ],
            SubjectType.SCIENCE: [
                'formula_checker', 'unit_converter', 'data_analyzer',
                'experiment_evaluator'
            ],
            SubjectType.HISTORY: [
                'fact_checker', 'timeline_analyzer', 'source_evaluator',
                'chronology_checker'
            ],
            SubjectType.GENERAL: [
                'grammar_checker', 'content_analyzer', 'factual_checker'
            ]
        }

    def classify_assignment(self, text: str, metadata: Dict[str, Any] = None) -> AssignmentClassification:
        """Classify assignment by subject, complexity, and other factors."""

        # Detect language first
        language_info = detect_text_language(text)
        detected_language = language_info.fallback_language if language_info else 'en'

        # Score subjects based on pattern matching
        subject_scores = {}
        text_lower = text.lower()

        for subject, patterns in self.subject_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            subject_scores[subject] = score

        # Consider metadata hints
        if metadata:
            subject_hint = metadata.get('subject', '').lower()
            class_hint = metadata.get('class', '').lower()

            if any(math_term in subject_hint + class_hint for math_term in ['math', 'algebra', 'calculus', 'geometry']):
                subject_scores[SubjectType.MATHEMATICS] = subject_scores.get(SubjectType.MATHEMATICS, 0) + 5

            if any(spanish_term in subject_hint + class_hint for spanish_term in ['spanish', 'español', 'language']):
                subject_scores[SubjectType.SPANISH] = subject_scores.get(SubjectType.SPANISH, 0) + 5

        # Determine primary subject
        best_subject = max(subject_scores.items(), key=lambda x: x[1]) if subject_scores else (SubjectType.UNKNOWN, 0)
        primary_subject = best_subject[0] if best_subject[1] > 0 else SubjectType.GENERAL
        confidence = min(best_subject[1] / 10.0, 1.0) if best_subject[1] > 0 else 0.1

        # Determine complexity
        complexity_scores = {}
        for complexity, patterns in self.complexity_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            complexity_scores[complexity] = score

        best_complexity = max(complexity_scores.items(), key=lambda x: x[1]) if complexity_scores else (AssignmentComplexity.UNKNOWN, 0)
        assignment_complexity = best_complexity[0] if best_complexity[1] > 0 else AssignmentComplexity.UNKNOWN

        # Determine specific type
        specific_type = self._determine_specific_type(primary_subject, text)

        # Determine tools needed
        tools_needed = self.subject_tools.get(primary_subject, self.subject_tools[SubjectType.GENERAL])

        # Determine processing approach
        processing_approach = self._determine_processing_approach(primary_subject, specific_type)

        return AssignmentClassification(
            subject=primary_subject,
            complexity=assignment_complexity,
            specific_type=specific_type,
            confidence=confidence,
            language=detected_language,
            tools_needed=tools_needed,
            processing_approach=processing_approach
        )

    def _determine_specific_type(self, subject: SubjectType, text: str) -> str:
        """Determine the specific assignment type within a subject."""
        text_lower = text.lower()

        if subject == SubjectType.MATHEMATICS:
            # Use math processor to determine specific type
            math_type = self.math_processor.identify_problem_type(text)
            return math_type.value

        elif subject == SubjectType.SPANISH:
            # Use Spanish processor to determine specific type
            spanish_type = self.spanish_processor.identify_assignment_type(text)
            return spanish_type.value

        elif subject == SubjectType.ENGLISH:
            if any(term in text_lower for term in ['essay', 'composition', 'write']):
                return 'writing'
            elif any(term in text_lower for term in ['analyze', 'analysis', 'interpret']):
                return 'analysis'
            elif any(term in text_lower for term in ['grammar', 'sentence', 'punctuation']):
                return 'grammar'
            else:
                return 'general'

        elif subject == SubjectType.SCIENCE:
            # Use science processor to determine specific type
            science_type = self.science_processor.identify_assignment_type(text)
            return science_type.value

        elif subject == SubjectType.HISTORY:
            # Use history processor to determine specific type
            history_type = self.history_processor.identify_assignment_type(text)
            return history_type.value

        return 'general'

    def _determine_processing_approach(self, subject: SubjectType, specific_type: str) -> str:
        """Determine the best processing approach for the assignment."""

        if subject == SubjectType.MATHEMATICS:
            if specific_type in ['algebra', 'calculus']:
                return 'symbolic_computation'
            elif specific_type == 'word_problem':
                return 'natural_language_math'
            elif specific_type == 'geometry':
                return 'visual_spatial'
            else:
                return 'standard_math'

        elif subject == SubjectType.SPANISH:
            if specific_type in ['grammar', 'conjugation']:
                return 'linguistic_analysis'
            elif specific_type == 'translation':
                return 'translation_evaluation'
            elif specific_type == 'culture':
                return 'cultural_assessment'
            else:
                return 'language_proficiency'

        elif subject == SubjectType.ENGLISH:
            if specific_type == 'writing':
                return 'composition_analysis'
            elif specific_type == 'analysis':
                return 'literary_analysis'
            else:
                return 'language_arts'

        elif subject == SubjectType.SCIENCE:
            if specific_type in ['laboratory_report', 'experimental_design']:
                return 'experimental_analysis'
            elif specific_type == 'data_analysis':
                return 'quantitative_analysis'
            elif specific_type == 'theoretical_explanation':
                return 'conceptual_analysis'
            else:
                return 'scientific_method'

        elif subject == SubjectType.HISTORY:
            if specific_type in ['chronological_analysis', 'timeline_creation']:
                return 'temporal_analysis'
            elif specific_type in ['cause_and_effect', 'historical_argument']:
                return 'causal_analysis'
            elif specific_type == 'source_analysis':
                return 'primary_source_evaluation'
            else:
                return 'historical_investigation'

        return 'general_assessment'

    async def process_assignment(self, text: str, source_text: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process assignment using the appropriate specialized processor."""

        # Classify the assignment
        classification = self.classify_assignment(text, metadata)

        result = {
            'classification': {
                'subject': classification.subject.value,
                'complexity': classification.complexity.value,
                'specific_type': classification.specific_type,
                'confidence': classification.confidence,
                'language': classification.language,
                'tools_needed': classification.tools_needed,
                'processing_approach': classification.processing_approach
            },
            'processing_results': {},
            'overall_score': 0.0,
            'specialized_feedback': [],
            'recommended_next_steps': []
        }

        # Route to appropriate processor
        try:
            if classification.subject == SubjectType.MATHEMATICS:
                math_results = self.math_processor.grade_math_assignment(text)
                result['processing_results'] = math_results
                result['overall_score'] = math_results['overall_score']
                result['specialized_feedback'] = math_results['feedback']

            elif classification.subject == SubjectType.SPANISH:
                spanish_results = self.spanish_processor.grade_spanish_assignment(text, source_text)
                result['processing_results'] = spanish_results
                result['overall_score'] = spanish_results['overall_score']
                result['specialized_feedback'] = spanish_results['feedback']

            elif classification.subject == SubjectType.SCIENCE:
                science_results = await self.science_processor.grade_science_assignment(text, source_text)
                result['processing_results'] = science_results
                result['overall_score'] = science_results['overall_score']
                result['specialized_feedback'] = science_results['feedback']

            elif classification.subject == SubjectType.HISTORY:
                history_results = await self.history_processor.grade_history_assignment(text, source_text)
                result['processing_results'] = history_results
                result['overall_score'] = history_results['overall_score']
                result['specialized_feedback'] = history_results['feedback']

            else:
                # Fall back to general processing
                result['processing_results'] = {
                    'message': f'General processing for {classification.subject.value} assignments',
                    'requires_specialized_processor': True
                }
                result['overall_score'] = 7.0  # Default moderate score
                result['specialized_feedback'] = [
                    f'This appears to be a {classification.subject.value} assignment.',
                    f'Consider implementing specialized processing for {classification.specific_type} assignments.'
                ]

        except Exception as e:
            result['processing_results'] = {
                'error': f'Error in specialized processing: {str(e)}',
                'fallback_used': True
            }
            result['overall_score'] = 5.0
            result['specialized_feedback'] = ['Unable to complete specialized analysis.']

        # Add recommended next steps
        result['recommended_next_steps'] = self._generate_next_steps(classification, result['overall_score'])

        return result

    def _generate_next_steps(self, classification: AssignmentClassification, score: float) -> List[str]:
        """Generate recommended next steps based on classification and performance."""
        steps = []

        if score < 6.0:
            steps.append(f"Review fundamental concepts in {classification.subject.value}")

        if classification.subject == SubjectType.MATHEMATICS:
            if classification.specific_type in ['algebra', 'calculus']:
                steps.append("Practice more equation-solving problems")
            elif classification.specific_type == 'geometry':
                steps.append("Work on visual-spatial reasoning exercises")

        elif classification.subject == SubjectType.SPANISH:
            if classification.specific_type == 'grammar':
                steps.append("Focus on verb conjugation and gender agreement")
            elif classification.specific_type == 'vocabulary':
                steps.append("Expand vocabulary with reading practice")

        elif classification.subject == SubjectType.SCIENCE:
            if classification.specific_type == 'laboratory_report':
                steps.append("Include proper experimental procedure and data analysis")
            elif classification.specific_type == 'hypothesis_testing':
                steps.append("Ensure hypothesis is testable and clearly stated")
            elif classification.specific_type == 'data_analysis':
                steps.append("Use appropriate graphs and statistical analysis")

        elif classification.subject == SubjectType.HISTORY:
            if classification.specific_type == 'chronological_analysis':
                steps.append("Include specific dates and proper chronological order")
            elif classification.specific_type == 'source_analysis':
                steps.append("Evaluate source reliability and historical context")
            elif classification.specific_type == 'cause_and_effect':
                steps.append("Clearly explain causal relationships between events")

        if classification.complexity == AssignmentComplexity.ELEMENTARY:
            steps.append("Build foundational skills before advancing")
        elif classification.complexity == AssignmentComplexity.COLLEGE:
            steps.append("Consider advanced research and analysis techniques")

        if not steps:
            steps.append("Continue practicing to maintain proficiency")

        return steps

    def get_available_processors(self) -> Dict[str, List[str]]:
        """Get information about available specialized processors."""
        return {
            'subjects': [subject.value for subject in SubjectType],
            'math_problem_types': [ptype.value for ptype in MathProblemType],
            'spanish_assignment_types': [atype.value for atype in SpanishAssignmentType],
            'science_assignment_types': [atype.value for atype in ScienceAssignmentType],
            'history_assignment_types': [atype.value for atype in HistoryAssignmentType],
            'complexity_levels': [level.value for level in AssignmentComplexity],
            'available_tools': {
                subject.value: tools for subject, tools in self.subject_tools.items()
            }
        }

def create_assignment_orchestrator() -> AssignmentOrchestrator:
    """Factory function to create an assignment orchestrator instance."""
    return AssignmentOrchestrator()