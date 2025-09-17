"""
Science Assignment Processor
Specialized processor for science assignments including physics, chemistry, biology, and general science.
Handles scientific method, data analysis, experimental design, and scientific accuracy assessment.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass
import asyncio

from llms import llm_manager

class ScienceSubject(Enum):
    """Science subject types."""
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    EARTH_SCIENCE = "earth_science"
    ENVIRONMENTAL_SCIENCE = "environmental_science"
    GENERAL_SCIENCE = "general_science"

class ScienceAssignmentType(Enum):
    """Types of science assignments."""
    LABORATORY_REPORT = "laboratory_report"
    EXPERIMENTAL_DESIGN = "experimental_design"
    DATA_ANALYSIS = "data_analysis"
    THEORETICAL_EXPLANATION = "theoretical_explanation"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    OBSERVATION_REPORT = "observation_report"
    SCIENTIFIC_RESEARCH = "scientific_research"
    PROBLEM_SOLVING = "problem_solving"
    GENERAL = "general"

@dataclass
class ScienceAnalysis:
    """Science assignment analysis results."""
    subject_area: ScienceSubject
    assignment_type: ScienceAssignmentType
    scientific_method_elements: Dict[str, bool]
    units_and_measurements: List[str]
    formulas_identified: List[str]
    data_tables_present: bool
    graphs_charts_present: bool
    hypothesis_present: bool
    conclusion_present: bool
    scientific_vocabulary_score: float
    experimental_variables: Dict[str, List[str]]
    safety_considerations: List[str]

class ScienceProcessor:
    """Specialized processor for science assignments."""

    def __init__(self):
        # Science subject patterns
        self.subject_patterns = {
            ScienceSubject.PHYSICS: [
                r'\b(?:force|energy|motion|velocity|acceleration|momentum|friction)\b',
                r'\b(?:wave|frequency|amplitude|electromagnetic|radiation)\b',
                r'\b(?:newton|joule|watt|hertz|coulomb|volt|ampere)\b',
                r'\b(?:gravity|electric|magnetic|quantum|relativity)\b',
                r'F\s*=\s*ma|E\s*=\s*mc²|v\s*=\s*d/t'
            ],
            ScienceSubject.CHEMISTRY: [
                r'\b(?:atom|molecule|element|compound|reaction|bond)\b',
                r'\b(?:acid|base|pH|molarity|concentration|solution)\b',
                r'\b(?:catalyst|enzyme|oxidation|reduction|equilibrium)\b',
                r'\b(?:hydrogen|oxygen|carbon|sodium|chlorine)\b',
                r'H2O|CO2|NaCl|C6H12O6|[A-Z][a-z]?\d*'
            ],
            ScienceSubject.BIOLOGY: [
                r'\b(?:cell|organism|species|evolution|genetics|DNA|RNA)\b',
                r'\b(?:ecosystem|photosynthesis|respiration|metabolism)\b',
                r'\b(?:bacteria|virus|fungus|plant|animal|protein)\b',
                r'\b(?:mitosis|meiosis|chromosome|gene|allele)\b',
                r'\b(?:biodiversity|adaptation|natural selection)\b'
            ],
            ScienceSubject.EARTH_SCIENCE: [
                r'\b(?:geology|weather|climate|earthquake|volcano)\b',
                r'\b(?:rock|mineral|fossil|sediment|erosion)\b',
                r'\b(?:atmosphere|ocean|continent|plate tectonics)\b',
                r'\b(?:solar system|planet|star|galaxy|universe)\b'
            ],
            ScienceSubject.ENVIRONMENTAL_SCIENCE: [
                r'\b(?:pollution|conservation|sustainability|ecosystem)\b',
                r'\b(?:renewable|fossil fuel|greenhouse|carbon footprint)\b',
                r'\b(?:biodiversity|habitat|endangered|extinction)\b',
                r'\b(?:recycling|waste|environmental impact)\b'
            ]
        }

        # Assignment type patterns
        self.assignment_patterns = {
            ScienceAssignmentType.LABORATORY_REPORT: [
                r'\b(?:lab|laboratory|experiment|procedure|materials|results)\b',
                r'\b(?:hypothesis|method|observation|conclusion|data)\b',
                r'\b(?:equipment|apparatus|setup|trial|measurement)\b'
            ],
            ScienceAssignmentType.EXPERIMENTAL_DESIGN: [
                r'\b(?:design|plan|control|variable|independent|dependent)\b',
                r'\b(?:hypothesis|prediction|test|procedure|method)\b',
                r'\b(?:control group|experimental group|variable)\b'
            ],
            ScienceAssignmentType.DATA_ANALYSIS: [
                r'\b(?:data|graph|chart|table|statistics|analysis)\b',
                r'\b(?:trend|pattern|correlation|average|standard deviation)\b',
                r'\b(?:interpret|analyze|compare|conclude)\b'
            ],
            ScienceAssignmentType.THEORETICAL_EXPLANATION: [
                r'\b(?:explain|theory|principle|law|concept|mechanism)\b',
                r'\b(?:why|how|cause|effect|reason|process)\b',
                r'\b(?:describe|discuss|compare|contrast)\b'
            ]
        }

        # Scientific method elements
        self.scientific_method_keywords = {
            'question': [r'\b(?:question|problem|inquiry|ask|wonder)\b'],
            'hypothesis': [r'\b(?:hypothesis|predict|expect|if.*then|propose)\b'],
            'materials': [r'\b(?:materials|equipment|apparatus|tools|supplies)\b'],
            'procedure': [r'\b(?:procedure|method|steps|process|protocol)\b'],
            'observations': [r'\b(?:observe|observation|data|record|measure)\b'],
            'results': [r'\b(?:results|findings|data|measurements|outcomes)\b'],
            'conclusion': [r'\b(?:conclusion|conclude|summarize|interpret|analyze)\b'],
            'discussion': [r'\b(?:discuss|explain|analyze|evaluate|implications)\b']
        }

        # Common scientific units
        self.unit_patterns = [
            r'\b\d+\s*(?:m|cm|mm|km|g|kg|mg|L|mL|s|min|h|°C|°F|K)\b',
            r'\b\d+\s*(?:N|J|W|Hz|V|A|Ω|Pa|atm|mol|M)\b',
            r'\b\d+\.?\d*\s*×\s*10\^?[-+]?\d+\b'  # Scientific notation
        ]

        # Formula patterns
        self.formula_patterns = [
            r'[A-Z][a-z]?\d*[+-]?\s*→\s*[A-Z][a-z]?\d*[+-]?',  # Chemical equations
            r'[A-Za-z]+\s*=\s*[A-Za-z0-9+\-*/().\s]+',  # Mathematical formulas
            r'F\s*=\s*ma|E\s*=\s*mc²|v\s*=\s*d/t|PV\s*=\s*nRT'  # Common physics formulas
        ]

    def identify_science_subject(self, text: str) -> ScienceSubject:
        """Identify the specific science subject area."""
        text_lower = text.lower()
        subject_scores = {}

        for subject, patterns in self.subject_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            subject_scores[subject] = score

        if not subject_scores or max(subject_scores.values()) == 0:
            return ScienceSubject.GENERAL_SCIENCE

        return max(subject_scores.items(), key=lambda x: x[1])[0]

    def identify_assignment_type(self, text: str) -> ScienceAssignmentType:
        """Identify the type of science assignment."""
        text_lower = text.lower()
        type_scores = {}

        for assignment_type, patterns in self.assignment_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            type_scores[assignment_type] = score

        if not type_scores or max(type_scores.values()) == 0:
            return ScienceAssignmentType.GENERAL

        return max(type_scores.items(), key=lambda x: x[1])[0]

    def analyze_scientific_method(self, text: str) -> Dict[str, bool]:
        """Analyze presence of scientific method elements."""
        text_lower = text.lower()
        method_elements = {}

        for element, patterns in self.scientific_method_keywords.items():
            found = False
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    found = True
                    break
            method_elements[element] = found

        return method_elements

    def extract_units_and_measurements(self, text: str) -> List[str]:
        """Extract units and measurements from the text."""
        units_found = []

        for pattern in self.unit_patterns:
            matches = re.findall(pattern, text)
            units_found.extend(matches)

        return list(set(units_found))  # Remove duplicates

    def identify_formulas(self, text: str) -> List[str]:
        """Identify scientific formulas in the text."""
        formulas_found = []

        for pattern in self.formula_patterns:
            matches = re.findall(pattern, text)
            formulas_found.extend(matches)

        return formulas_found

    def analyze_data_visualization(self, text: str) -> Tuple[bool, bool]:
        """Check for presence of data tables and graphs/charts."""
        text_lower = text.lower()

        table_indicators = [
            r'\b(?:table|data.*table|results.*table)\b',
            r'\|.*\|.*\|',  # Table-like structure
            r'\b(?:row|column|cell)\b'
        ]

        graph_indicators = [
            r'\b(?:graph|chart|plot|figure|diagram)\b',
            r'\b(?:x-axis|y-axis|scatter|bar chart|line graph)\b',
            r'\b(?:histogram|pie chart|box plot)\b'
        ]

        has_tables = any(re.search(pattern, text_lower) for pattern in table_indicators)
        has_graphs = any(re.search(pattern, text_lower) for pattern in graph_indicators)

        return has_tables, has_graphs

    def assess_scientific_vocabulary(self, text: str, subject: ScienceSubject) -> float:
        """Assess the use of appropriate scientific vocabulary."""
        text_lower = text.lower()
        total_words = len(text.split())

        if total_words == 0:
            return 0.0

        # General scientific terms
        scientific_terms = [
            'hypothesis', 'theory', 'experiment', 'data', 'observation',
            'analysis', 'conclusion', 'variable', 'control', 'method',
            'evidence', 'phenomenon', 'principle', 'law', 'model'
        ]

        # Add subject-specific terms
        if subject in self.subject_patterns:
            for pattern in self.subject_patterns[subject]:
                # Extract words from regex patterns
                words = re.findall(r'\w+', pattern)
                scientific_terms.extend(words)

        scientific_word_count = 0
        for term in scientific_terms:
            scientific_word_count += len(re.findall(rf'\b{term}\b', text_lower))

        # Calculate vocabulary score (0-10 scale)
        vocabulary_ratio = scientific_word_count / total_words
        return min(vocabulary_ratio * 50, 10.0)  # Scale to 0-10

    def identify_experimental_variables(self, text: str) -> Dict[str, List[str]]:
        """Identify experimental variables mentioned in the text."""
        text_lower = text.lower()
        variables = {
            'independent': [],
            'dependent': [],
            'controlled': []
        }

        # Patterns for different variable types
        independent_patterns = [
            r'independent\s+variable[:\s]*([^.]*)',
            r'manipulated\s+variable[:\s]*([^.]*)',
            r'what\s+(?:we|you|i)\s+change[d]?[:\s]*([^.]*)'
        ]

        dependent_patterns = [
            r'dependent\s+variable[:\s]*([^.]*)',
            r'responding\s+variable[:\s]*([^.]*)',
            r'what\s+(?:we|you|i)\s+measure[d]?[:\s]*([^.]*)'
        ]

        controlled_patterns = [
            r'controlled?\s+variable[s]?[:\s]*([^.]*)',
            r'constant[s]?[:\s]*([^.]*)',
            r'keep\s+(?:the\s+)?same[:\s]*([^.]*)'
        ]

        for pattern in independent_patterns:
            matches = re.findall(pattern, text_lower)
            variables['independent'].extend([match.strip() for match in matches])

        for pattern in dependent_patterns:
            matches = re.findall(pattern, text_lower)
            variables['dependent'].extend([match.strip() for match in matches])

        for pattern in controlled_patterns:
            matches = re.findall(pattern, text_lower)
            variables['controlled'].extend([match.strip() for match in matches])

        return variables

    def identify_safety_considerations(self, text: str) -> List[str]:
        """Identify safety considerations mentioned in the assignment."""
        text_lower = text.lower()
        safety_terms = [
            'safety', 'caution', 'warning', 'protective equipment', 'goggles',
            'gloves', 'ventilation', 'fume hood', 'disposal', 'hazard',
            'toxic', 'flammable', 'corrosive', 'first aid'
        ]

        safety_found = []
        for term in safety_terms:
            if term in text_lower:
                safety_found.append(term)

        return safety_found

    def analyze_science_assignment(self, text: str) -> ScienceAnalysis:
        """Perform comprehensive analysis of science assignment."""
        subject_area = self.identify_science_subject(text)
        assignment_type = self.identify_assignment_type(text)

        scientific_method_elements = self.analyze_scientific_method(text)
        units_and_measurements = self.extract_units_and_measurements(text)
        formulas_identified = self.identify_formulas(text)

        has_tables, has_graphs = self.analyze_data_visualization(text)

        hypothesis_present = scientific_method_elements.get('hypothesis', False)
        conclusion_present = scientific_method_elements.get('conclusion', False)

        scientific_vocabulary_score = self.assess_scientific_vocabulary(text, subject_area)
        experimental_variables = self.identify_experimental_variables(text)
        safety_considerations = self.identify_safety_considerations(text)

        return ScienceAnalysis(
            subject_area=subject_area,
            assignment_type=assignment_type,
            scientific_method_elements=scientific_method_elements,
            units_and_measurements=units_and_measurements,
            formulas_identified=formulas_identified,
            data_tables_present=has_tables,
            graphs_charts_present=has_graphs,
            hypothesis_present=hypothesis_present,
            conclusion_present=conclusion_present,
            scientific_vocabulary_score=scientific_vocabulary_score,
            experimental_variables=experimental_variables,
            safety_considerations=safety_considerations
        )

    async def grade_science_assignment(self, text: str, source_text: str = None) -> Dict[str, Any]:
        """Grade a science assignment with specialized criteria."""
        analysis = self.analyze_science_assignment(text)

        # Create grading prompt based on assignment type and subject
        grading_prompt = self._create_science_grading_prompt(text, analysis, source_text)

        try:
            if llm_manager:
                response = llm_manager.invoke_with_fallback(
                    grading_prompt,
                    use_case="science_analysis"
                )
                llm_feedback = response.content
            else:
                llm_feedback = "LLM not available for detailed grading."
        except Exception as e:
            llm_feedback = f"Error in LLM grading: {str(e)}"

        # Calculate specialized scores
        scores = self._calculate_science_scores(analysis, text)

        return {
            'subject_area': analysis.subject_area.value,
            'assignment_type': analysis.assignment_type.value,
            'analysis': {
                'scientific_method_elements': analysis.scientific_method_elements,
                'units_and_measurements_count': len(analysis.units_and_measurements),
                'formulas_identified_count': len(analysis.formulas_identified),
                'data_visualization_present': analysis.data_tables_present or analysis.graphs_charts_present,
                'hypothesis_present': analysis.hypothesis_present,
                'conclusion_present': analysis.conclusion_present,
                'scientific_vocabulary_score': analysis.scientific_vocabulary_score,
                'experimental_variables': analysis.experimental_variables,
                'safety_considerations_count': len(analysis.safety_considerations)
            },
            'grading': scores,
            'overall_score': sum(scores.values()) / len(scores) if scores else 0.0,
            'feedback': [
                f"Science Subject: {analysis.subject_area.value.title()}",
                f"Assignment Type: {analysis.assignment_type.value.replace('_', ' ').title()}",
                f"Scientific Vocabulary Score: {analysis.scientific_vocabulary_score:.1f}/10",
                f"Scientific Method Elements: {sum(analysis.scientific_method_elements.values())}/{len(analysis.scientific_method_elements)}",
                llm_feedback
            ],
            'specialized_insights': self._generate_science_insights(analysis),
            'recommendations': self._generate_science_recommendations(analysis, scores)
        }

    def _create_science_grading_prompt(self, text: str, analysis: ScienceAnalysis, source_text: str = None) -> str:
        """Create a specialized grading prompt for science assignments."""

        base_prompt = f"""
        Please grade this {analysis.subject_area.value} assignment of type {analysis.assignment_type.value}.

        Assignment Text:
        {text}

        Evaluate the assignment on these science-specific criteria (0-10 scale):

        1. Scientific Accuracy: Are the facts, formulas, and concepts correct?
        2. Hypothesis Quality: Is there a clear, testable hypothesis (if applicable)?
        3. Data Analysis: How well is data presented, interpreted, and analyzed?
        4. Experimental Design: How well designed is the experiment or study?
        5. Conclusion Validity: Are conclusions supported by evidence and logical?

        Consider the following analysis:
        - Scientific Method Elements Present: {sum(analysis.scientific_method_elements.values())}/{len(analysis.scientific_method_elements)}
        - Units/Measurements Found: {len(analysis.units_and_measurements)}
        - Formulas Identified: {len(analysis.formulas_identified)}
        - Data Visualization: {'Yes' if analysis.data_tables_present or analysis.graphs_charts_present else 'No'}
        - Scientific Vocabulary Score: {analysis.scientific_vocabulary_score:.1f}/10

        Provide specific feedback on scientific reasoning, methodology, and presentation.
        """

        if source_text:
            base_prompt += f"\n\nReference Material:\n{source_text}\n\nAlso evaluate how well the assignment addresses the source material."

        return base_prompt

    def _calculate_science_scores(self, analysis: ScienceAnalysis, text: str) -> Dict[str, float]:
        """Calculate specialized science scores."""
        scores = {}

        # Scientific Accuracy (estimated based on vocabulary and structure)
        scores['scientific_accuracy'] = min(analysis.scientific_vocabulary_score + 1.0, 10.0)

        # Hypothesis Quality
        if analysis.hypothesis_present:
            # Check for qualities of a good hypothesis
            hypothesis_quality = 7.0
            if any(word in text.lower() for word in ['if', 'then', 'because', 'predict']):
                hypothesis_quality += 1.0
            if any(word in text.lower() for word in ['testable', 'measurable', 'specific']):
                hypothesis_quality += 1.0
            scores['hypothesis_quality'] = min(hypothesis_quality, 10.0)
        else:
            scores['hypothesis_quality'] = 3.0 if analysis.assignment_type in [
                ScienceAssignmentType.THEORETICAL_EXPLANATION,
                ScienceAssignmentType.OBSERVATION_REPORT
            ] else 0.0

        # Data Analysis
        data_score = 5.0  # Base score
        if analysis.data_tables_present:
            data_score += 2.0
        if analysis.graphs_charts_present:
            data_score += 2.0
        if len(analysis.units_and_measurements) > 0:
            data_score += 1.0
        scores['data_analysis'] = min(data_score, 10.0)

        # Experimental Design
        design_score = 5.0
        method_elements = sum(analysis.scientific_method_elements.values())
        design_score += (method_elements / len(analysis.scientific_method_elements)) * 3.0

        if len(analysis.experimental_variables['independent']) > 0:
            design_score += 1.0
        if len(analysis.experimental_variables['dependent']) > 0:
            design_score += 1.0

        scores['experimental_design'] = min(design_score, 10.0)

        # Conclusion Validity
        if analysis.conclusion_present:
            conclusion_score = 7.0
            if 'evidence' in text.lower() or 'data' in text.lower():
                conclusion_score += 1.0
            if any(word in text.lower() for word in ['therefore', 'thus', 'consequently', 'based on']):
                conclusion_score += 1.0
            if any(word in text.lower() for word in ['support', 'confirm', 'reject', 'accept']):
                conclusion_score += 1.0
            scores['conclusion_validity'] = min(conclusion_score, 10.0)
        else:
            scores['conclusion_validity'] = 3.0

        return scores

    def _generate_science_insights(self, analysis: ScienceAnalysis) -> List[str]:
        """Generate insights specific to science assignments."""
        insights = []

        insights.append(f"Assignment focuses on {analysis.subject_area.value.replace('_', ' ')}")
        insights.append(f"Type: {analysis.assignment_type.value.replace('_', ' ')}")

        if analysis.formulas_identified:
            insights.append(f"Contains {len(analysis.formulas_identified)} scientific formulas/equations")

        if analysis.units_and_measurements:
            insights.append(f"Uses {len(analysis.units_and_measurements)} different units/measurements")

        if analysis.experimental_variables['independent']:
            insights.append(f"Identifies independent variables: {', '.join(analysis.experimental_variables['independent'])}")

        if analysis.safety_considerations:
            insights.append(f"Addresses {len(analysis.safety_considerations)} safety considerations")

        # Scientific method completeness
        method_score = sum(analysis.scientific_method_elements.values())
        total_elements = len(analysis.scientific_method_elements)
        insights.append(f"Scientific method completeness: {method_score}/{total_elements} elements present")

        return insights

    def _generate_science_recommendations(self, analysis: ScienceAnalysis, scores: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving science assignments."""
        recommendations = []

        # Hypothesis recommendations
        if scores.get('hypothesis_quality', 0) < 6.0:
            if not analysis.hypothesis_present:
                recommendations.append("Include a clear, testable hypothesis")
            else:
                recommendations.append("Make hypothesis more specific and measurable")

        # Data analysis recommendations
        if scores.get('data_analysis', 0) < 6.0:
            if not analysis.data_tables_present:
                recommendations.append("Include data tables to organize measurements")
            if not analysis.graphs_charts_present:
                recommendations.append("Add graphs or charts to visualize data trends")

        # Experimental design recommendations
        if scores.get('experimental_design', 0) < 6.0:
            missing_elements = [k for k, v in analysis.scientific_method_elements.items() if not v]
            if missing_elements:
                recommendations.append(f"Include missing scientific method elements: {', '.join(missing_elements)}")

        # Variable identification
        if not analysis.experimental_variables['independent']:
            recommendations.append("Clearly identify independent (manipulated) variables")
        if not analysis.experimental_variables['dependent']:
            recommendations.append("Clearly identify dependent (responding) variables")

        # Scientific vocabulary
        if analysis.scientific_vocabulary_score < 6.0:
            recommendations.append(f"Use more {analysis.subject_area.value.replace('_', ' ')} vocabulary and terminology")

        # Safety considerations
        if len(analysis.safety_considerations) == 0 and analysis.assignment_type == ScienceAssignmentType.LABORATORY_REPORT:
            recommendations.append("Include safety considerations and precautions")

        # Units and measurements
        if len(analysis.units_and_measurements) == 0:
            recommendations.append("Include proper units with all measurements and calculations")

        return recommendations

def create_science_processor() -> ScienceProcessor:
    """Factory function to create a ScienceProcessor instance."""
    return ScienceProcessor()