"""
History Assignment Processor
Specialized processor for history assignments including chronological analysis, historical accuracy,
source evaluation, and contextual understanding assessment.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass
import asyncio
from datetime import datetime

from core.llms import llm_manager

class HistoryPeriod(Enum):
    """Historical time periods."""
    ANCIENT = "ancient"
    CLASSICAL = "classical"
    MEDIEVAL = "medieval"
    RENAISSANCE = "renaissance"
    EARLY_MODERN = "early_modern"
    INDUSTRIAL = "industrial"
    MODERN = "modern"
    CONTEMPORARY = "contemporary"
    UNKNOWN = "unknown"

class HistoryAssignmentType(Enum):
    """Types of history assignments."""
    CHRONOLOGICAL_ANALYSIS = "chronological_analysis"
    CAUSE_AND_EFFECT = "cause_and_effect"
    BIOGRAPHICAL_STUDY = "biographical_study"
    SOURCE_ANALYSIS = "source_analysis"
    COMPARATIVE_STUDY = "comparative_study"
    THEMATIC_ESSAY = "thematic_essay"
    RESEARCH_PAPER = "research_paper"
    TIMELINE_CREATION = "timeline_creation"
    HISTORICAL_ARGUMENT = "historical_argument"
    GENERAL = "general"

class RegionFocus(Enum):
    """Geographic/regional focus."""
    WORLD_HISTORY = "world_history"
    AMERICAN_HISTORY = "american_history"
    EUROPEAN_HISTORY = "european_history"
    ASIAN_HISTORY = "asian_history"
    AFRICAN_HISTORY = "african_history"
    LATIN_AMERICAN_HISTORY = "latin_american_history"
    MIDDLE_EASTERN_HISTORY = "middle_eastern_history"
    UNKNOWN = "unknown"

@dataclass
class HistoryAnalysis:
    """History assignment analysis results."""
    period: HistoryPeriod
    assignment_type: HistoryAssignmentType
    region_focus: RegionFocus
    dates_identified: List[str]
    historical_figures: List[str]
    events_mentioned: List[str]
    sources_cited: List[str]
    chronological_accuracy: float
    historical_context_score: float
    argument_structure_score: float
    evidence_usage_score: float
    bias_awareness_score: float
    historical_vocabulary_score: float

class HistoryProcessor:
    """Specialized processor for history assignments."""

    def __init__(self, llm_manager: Any = None):
        self.llm_manager = llm_manager
        # Historical period patterns
        self.period_patterns = {
            HistoryPeriod.ANCIENT: [
                r'\b(?:ancient|antiquity|prehistoric|stone age|bronze age|iron age)\b',
                r'\b(?:mesopotamia|egypt|greece|rome|babylon|assyria)\b',
                r'\b(?:pyramid|pharaoh|caesar|alexander|aristotle|plato)\b',
                r'\b(?:3000|2000|1000)\s*(?:BC|BCE)\b'
            ],
            HistoryPeriod.CLASSICAL: [
                r'\b(?:classical|greco-roman|hellenistic|republican|imperial)\b',
                r'\b(?:athens|sparta|roman empire|constantinople)\b',
                r'\b(?:democracy|republic|senate|emperor|gladiator)\b',
                r'\b(?:[1-5]\d{2})\s*(?:BC|BCE|AD|CE)\b'
            ],
            HistoryPeriod.MEDIEVAL: [
                r'\b(?:medieval|middle ages|feudal|crusades|byzantine)\b',
                r'\b(?:knight|castle|manor|serf|lord|vassal)\b',
                r'\b(?:charlemagne|holy roman empire|black death)\b',
                r'\b(?:[5-9]\d{2}|1[0-4]\d{2})\s*(?:AD|CE)\b'
            ],
            HistoryPeriod.RENAISSANCE: [
                r'\b(?:renaissance|reformation|humanism|enlightenment)\b',
                r'\b(?:leonardo|michelangelo|shakespeare|galileo|luther)\b',
                r'\b(?:protestant|catholic|printing press|exploration)\b',
                r'\b(?:1[4-6]\d{2})\s*(?:AD|CE)?\b'
            ],
            HistoryPeriod.EARLY_MODERN: [
                r'\b(?:colonial|exploration|conquest|empire|trading)\b',
                r'\b(?:columbus|cortez|magellan|jamestown|plymouth)\b',
                r'\b(?:mercantilism|absolute monarchy|scientific revolution)\b',
                r'\b(?:1[5-7]\d{2})\s*(?:AD|CE)?\b'
            ],
            HistoryPeriod.INDUSTRIAL: [
                r'\b(?:industrial revolution|factory|steam|railroad|urbanization)\b',
                r'\b(?:cotton gin|telegraph|immigration|labor|union)\b',
                r'\b(?:capitalism|socialism|imperialism|nationalism)\b',
                r'\b(?:1[7-9]\d{2})\s*(?:AD|CE)?\b'
            ],
            HistoryPeriod.MODERN: [
                r'\b(?:world war|depression|cold war|civil rights)\b',
                r'\b(?:hitler|stalin|roosevelt|kennedy|king)\b',
                r'\b(?:fascism|communism|democracy|nuclear|holocaust)\b',
                r'\b(?:19[0-6]\d|197\d)\b'
            ],
            HistoryPeriod.CONTEMPORARY: [
                r'\b(?:globalization|terrorism|internet|climate change)\b',
                r'\b(?:9/11|obama|brexit|covid|pandemic|social media)\b',
                r'\b(?:digital age|information age|modern era)\b',
                r'\b(?:19[8-9]\d|20[0-2]\d)\b'
            ]
        }

        # Assignment type patterns
        self.assignment_patterns = {
            HistoryAssignmentType.CHRONOLOGICAL_ANALYSIS: [
                r'\b(?:timeline|chronology|sequence|order|before|after)\b',
                r'\b(?:first|then|next|finally|meanwhile|simultaneously)\b',
                r'\b(?:when|date|year|period|era|time)\b'
            ],
            HistoryAssignmentType.CAUSE_AND_EFFECT: [
                r'\b(?:cause|effect|result|consequence|led to|resulted in)\b',
                r'\b(?:because|therefore|thus|why|reason|factor)\b',
                r'\b(?:influence|impact|outcome|aftermath)\b'
            ],
            HistoryAssignmentType.BIOGRAPHICAL_STUDY: [
                r'\b(?:biography|life|born|died|career|achievement)\b',
                r'\b(?:person|individual|leader|figure|character)\b',
                r'\b(?:childhood|education|family|legacy)\b'
            ],
            HistoryAssignmentType.SOURCE_ANALYSIS: [
                r'\b(?:source|document|primary|secondary|evidence)\b',
                r'\b(?:author|perspective|bias|reliability|credibility)\b',
                r'\b(?:analyze|interpret|evaluate|examine)\b'
            ],
            HistoryAssignmentType.COMPARATIVE_STUDY: [
                r'\b(?:compare|contrast|similar|different|alike)\b',
                r'\b(?:versus|vs|both|either|neither|whereas)\b',
                r'\b(?:similarity|difference|comparison)\b'
            ],
            HistoryAssignmentType.HISTORICAL_ARGUMENT: [
                r'\b(?:argue|argument|thesis|claim|position|stance)\b',
                r'\b(?:evidence|support|proof|demonstrate|show)\b',
                r'\b(?:persuade|convince|defend|justify)\b'
            ]
        }

        # Regional focus patterns
        self.region_patterns = {
            RegionFocus.AMERICAN_HISTORY: [
                r'\b(?:america|united states|usa|colonial|revolutionary)\b',
                r'\b(?:constitution|civil war|slavery|independence)\b',
                r'\b(?:washington|lincoln|jefferson|roosevelt)\b'
            ],
            RegionFocus.EUROPEAN_HISTORY: [
                r'\b(?:europe|britain|france|germany|italy|spain)\b',
                r'\b(?:england|british|french|german|italian|spanish)\b',
                r'\b(?:napoleon|hitler|churchill|elizabeth|victoria)\b'
            ],
            RegionFocus.ASIAN_HISTORY: [
                r'\b(?:asia|china|japan|india|korea|vietnam)\b',
                r'\b(?:chinese|japanese|indian|asian|confucius)\b',
                r'\b(?:dynasty|emperor|shogun|mandarin|samurai)\b'
            ],
            RegionFocus.AFRICAN_HISTORY: [
                r'\b(?:africa|african|egypt|ethiopia|ghana|mali)\b',
                r'\b(?:sahara|nile|slavery|apartheid|mandela)\b',
                r'\b(?:tribe|kingdom|colonization|independence)\b'
            ]
        }

        # Date patterns for extraction
        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b\d{4}\s*(?:BC|BCE|AD|CE)\b',        # Year with era
            r'\b(?:19|20)\d{2}\b',                   # 20th/21st century years
            r'\b(?:1[0-8]|[1-9])\d{2}\b',          # Earlier years
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]

        # Historical figures pattern (common names)
        self.historical_figures_patterns = [
            r'\b(?:Washington|Lincoln|Jefferson|Roosevelt|Kennedy|Reagan)\b',
            r'\b(?:Napoleon|Hitler|Stalin|Churchill|Elizabeth|Victoria)\b',
            r'\b(?:Caesar|Alexander|Cleopatra|Constantine|Charlemagne)\b',
            r'\b(?:Gandhi|Mandela|King|Malcolm|Parks|Anthony)\b',
            r'\b(?:Shakespeare|Leonardo|Michelangelo|Galileo|Newton)\b'
        ]

        # Historical vocabulary terms
        self.historical_vocabulary = [
            'revolution', 'empire', 'dynasty', 'civilization', 'culture',
            'society', 'government', 'politics', 'economy', 'religion',
            'military', 'warfare', 'diplomacy', 'treaty', 'alliance',
            'reform', 'movement', 'ideology', 'nationalism', 'imperialism',
            'colonialism', 'independence', 'democracy', 'monarchy', 'republic'
        ]

    def identify_historical_period(self, text: str) -> HistoryPeriod:
        """Identify the historical time period of the assignment."""
        text_lower = text.lower()
        period_scores = {}

        for period, patterns in self.period_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            period_scores[period] = score

        if not period_scores or max(period_scores.values()) == 0:
            return HistoryPeriod.UNKNOWN

        return max(period_scores.items(), key=lambda x: x[1])[0]

    def identify_assignment_type(self, text: str) -> HistoryAssignmentType:
        """Identify the type of history assignment."""
        text_lower = text.lower()
        type_scores = {}

        for assignment_type, patterns in self.assignment_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            type_scores[assignment_type] = score

        if not type_scores or max(type_scores.values()) == 0:
            return HistoryAssignmentType.GENERAL

        return max(type_scores.items(), key=lambda x: x[1])[0]

    def identify_region_focus(self, text: str) -> RegionFocus:
        """Identify the regional focus of the history assignment."""
        text_lower = text.lower()
        region_scores = {}

        for region, patterns in self.region_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            region_scores[region] = score

        if not region_scores or max(region_scores.values()) == 0:
            return RegionFocus.WORLD_HISTORY

        return max(region_scores.items(), key=lambda x: x[1])[0]

    def extract_dates(self, text: str) -> List[str]:
        """Extract dates and time references from the text."""
        dates_found = []

        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            dates_found.extend(matches)

        return list(set(dates_found))  # Remove duplicates

    def extract_historical_figures(self, text: str) -> List[str]:
        """Extract mentions of historical figures."""
        figures_found = []

        for pattern in self.historical_figures_patterns:
            matches = re.findall(pattern, text)
            figures_found.extend(matches)

        return list(set(figures_found))

    def extract_historical_events(self, text: str) -> List[str]:
        """Extract mentions of historical events."""
        text_lower = text.lower()
        events = []

        # Common historical events patterns
        event_patterns = [
            r'\b(?:world war|civil war|revolutionary war|cold war)\b',
            r'\b(?:depression|recession|economic crisis)\b',
            r'\b(?:revolution|reformation|renaissance|enlightenment)\b',
            r'\b(?:crusades|holocaust|genocide|massacre)\b',
            r'\b(?:independence|emancipation|suffrage|civil rights)\b',
            r'\b(?:battle of|siege of|treaty of|declaration of)\s+\w+\b'
        ]

        for pattern in event_patterns:
            matches = re.findall(pattern, text_lower)
            events.extend(matches)

        return list(set(events))

    def extract_sources(self, text: str) -> List[str]:
        """Extract source citations and references."""
        sources_found = []

        # Common citation patterns
        citation_patterns = [
            r'\b(?:according to|as stated by|source:|reference:)\s+([^.]+)',
            r'\([^)]*\d{4}[^)]*\)',  # Parenthetical citations with years
            r'\"[^\"]+\"\s*-\s*[A-Z][^.]+',  # Quoted material with attribution
            r'\b(?:document|letter|speech|diary|memoir|newspaper)\s+(?:by|from|of)\s+[A-Z][^.]+',
        ]

        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            sources_found.extend(matches)

        return sources_found

    def assess_chronological_accuracy(self, text: str, dates: List[str]) -> float:
        """Assess chronological accuracy and organization."""
        if not dates:
            return 5.0  # Neutral score if no dates found

        text_lower = text.lower()

        # Check for chronological indicators
        chronological_words = [
            'first', 'then', 'next', 'after', 'before', 'during',
            'meanwhile', 'simultaneously', 'later', 'earlier',
            'subsequently', 'preceding', 'following'
        ]

        chronological_score = 5.0
        for word in chronological_words:
            if word in text_lower:
                chronological_score += 0.5

        # Bonus for having multiple dates (suggests timeline awareness)
        if len(dates) > 1:
            chronological_score += 1.0

        return min(chronological_score, 10.0)

    def assess_historical_context(self, text: str, period: HistoryPeriod) -> float:
        """Assess understanding of historical context."""
        text_lower = text.lower()

        context_indicators = [
            'context', 'background', 'setting', 'circumstances',
            'conditions', 'environment', 'situation', 'climate',
            'social', 'political', 'economic', 'cultural'
        ]

        context_score = 3.0  # Base score
        for indicator in context_indicators:
            if indicator in text_lower:
                context_score += 0.5

        # Bonus for period-specific context awareness
        if period != HistoryPeriod.UNKNOWN:
            context_score += 1.0

        return min(context_score, 10.0)

    def assess_argument_structure(self, text: str) -> float:
        """Assess the structure and logic of historical arguments."""
        text_lower = text.lower()

        argument_indicators = [
            'thesis', 'argument', 'claim', 'position', 'stance',
            'evidence', 'proof', 'support', 'demonstrate', 'show',
            'therefore', 'thus', 'consequently', 'however', 'although',
            'furthermore', 'moreover', 'in addition', 'conclusion'
        ]

        structure_score = 3.0
        for indicator in argument_indicators:
            if indicator in text_lower:
                structure_score += 0.3

        # Check for paragraph structure (multiple paragraphs suggest organization)
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 2:
            structure_score += 1.0

        return min(structure_score, 10.0)

    def assess_evidence_usage(self, text: str, sources: List[str]) -> float:
        """Assess the use of historical evidence and sources."""
        text_lower = text.lower()

        evidence_score = 3.0

        # Check for evidence-related language
        evidence_words = [
            'evidence', 'source', 'document', 'record', 'account',
            'testimony', 'artifact', 'primary', 'secondary',
            'according to', 'as stated', 'reported', 'documented'
        ]

        for word in evidence_words:
            if word in text_lower:
                evidence_score += 0.4

        # Bonus for having sources
        if sources:
            evidence_score += len(sources) * 0.5

        return min(evidence_score, 10.0)

    def assess_bias_awareness(self, text: str) -> float:
        """Assess awareness of historical bias and perspective."""
        text_lower = text.lower()

        bias_indicators = [
            'bias', 'perspective', 'viewpoint', 'opinion', 'interpretation',
            'subjective', 'objective', 'point of view', 'angle',
            'may have', 'might be', 'could be', 'possibly', 'likely',
            'question', 'doubt', 'uncertain', 'debate', 'controversy'
        ]

        bias_score = 3.0
        for indicator in bias_indicators:
            if indicator in text_lower:
                bias_score += 0.5

        return min(bias_score, 10.0)

    def assess_historical_vocabulary(self, text: str) -> float:
        """Assess the use of appropriate historical vocabulary."""
        text_lower = text.lower()
        total_words = len(text.split())

        if total_words == 0:
            return 0.0

        historical_word_count = 0
        for term in self.historical_vocabulary:
            historical_word_count += len(re.findall(rf'\b{term}\b', text_lower))

        # Calculate vocabulary score (0-10 scale)
        vocabulary_ratio = historical_word_count / total_words
        return min(vocabulary_ratio * 40, 10.0)  # Scale to 0-10

    def analyze_history_assignment(self, text: str) -> HistoryAnalysis:
        """Perform comprehensive analysis of history assignment."""
        period = self.identify_historical_period(text)
        assignment_type = self.identify_assignment_type(text)
        region_focus = self.identify_region_focus(text)

        dates_identified = self.extract_dates(text)
        historical_figures = self.extract_historical_figures(text)
        events_mentioned = self.extract_historical_events(text)
        sources_cited = self.extract_sources(text)

        chronological_accuracy = self.assess_chronological_accuracy(text, dates_identified)
        historical_context_score = self.assess_historical_context(text, period)
        argument_structure_score = self.assess_argument_structure(text)
        evidence_usage_score = self.assess_evidence_usage(text, sources_cited)
        bias_awareness_score = self.assess_bias_awareness(text)
        historical_vocabulary_score = self.assess_historical_vocabulary(text)

        return HistoryAnalysis(
            period=period,
            assignment_type=assignment_type,
            region_focus=region_focus,
            dates_identified=dates_identified,
            historical_figures=historical_figures,
            events_mentioned=events_mentioned,
            sources_cited=sources_cited,
            chronological_accuracy=chronological_accuracy,
            historical_context_score=historical_context_score,
            argument_structure_score=argument_structure_score,
            evidence_usage_score=evidence_usage_score,
            bias_awareness_score=bias_awareness_score,
            historical_vocabulary_score=historical_vocabulary_score
        )

    async def grade_history_assignment(self, text: str, source_text: str = None) -> Dict[str, Any]:
        """Grade a history assignment with specialized criteria."""
        analysis = self.analyze_history_assignment(text)

        # Create grading prompt based on assignment type and period
        grading_prompt = self._create_history_grading_prompt(text, analysis, source_text)

        try:
            if llm_manager:
                response = llm_manager.invoke_with_fallback(
                    grading_prompt,
                    use_case="history_analysis"
                )
                llm_feedback = response.content
            else:
                llm_feedback = "LLM not available for detailed grading."
        except Exception as e:
            llm_feedback = f"Error in LLM grading: {str(e)}"

        # Calculate specialized scores
        scores = self._calculate_history_scores(analysis)

        return {
            'period': analysis.period.value,
            'assignment_type': analysis.assignment_type.value,
            'region_focus': analysis.region_focus.value,
            'analysis': {
                'dates_identified_count': len(analysis.dates_identified),
                'historical_figures_count': len(analysis.historical_figures),
                'events_mentioned_count': len(analysis.events_mentioned),
                'sources_cited_count': len(analysis.sources_cited),
                'chronological_accuracy': analysis.chronological_accuracy,
                'historical_context_score': analysis.historical_context_score,
                'argument_structure_score': analysis.argument_structure_score,
                'evidence_usage_score': analysis.evidence_usage_score,
                'bias_awareness_score': analysis.bias_awareness_score,
                'historical_vocabulary_score': analysis.historical_vocabulary_score
            },
            'grading': scores,
            'overall_score': sum(scores.values()) / len(scores) if scores else 0.0,
            'feedback': [
                f"Historical Period: {analysis.period.value.title()}",
                f"Assignment Type: {analysis.assignment_type.value.replace('_', ' ').title()}",
                f"Regional Focus: {analysis.region_focus.value.replace('_', ' ').title()}",
                f"Historical Vocabulary Score: {analysis.historical_vocabulary_score:.1f}/10",
                f"Sources Referenced: {len(analysis.sources_cited)}",
                llm_feedback
            ],
            'specialized_insights': self._generate_history_insights(analysis),
            'recommendations': self._generate_history_recommendations(analysis, scores)
        }

    def _create_history_grading_prompt(self, text: str, analysis: HistoryAnalysis, source_text: str = None) -> str:
        """Create a specialized grading prompt for history assignments."""

        base_prompt = f"""
        Please grade this {analysis.period.value} history assignment of type {analysis.assignment_type.value}.

        Assignment Text:
        {text}

        Evaluate the assignment on these history-specific criteria (0-10 scale):

        1. Historical Accuracy: Are facts, dates, and events presented correctly?
        2. Chronological Understanding: How well does the student understand sequence and timing?
        3. Source Analysis: How effectively are sources used and evaluated?
        4. Contextual Awareness: Does the student understand the historical context?
        5. Argument Development: How well-structured and supported are historical arguments?

        Consider the following analysis:
        - Historical Period: {analysis.period.value}
        - Regional Focus: {analysis.region_focus.value}
        - Dates Identified: {len(analysis.dates_identified)}
        - Historical Figures: {len(analysis.historical_figures)}
        - Events Mentioned: {len(analysis.events_mentioned)}
        - Sources Cited: {len(analysis.sources_cited)}
        - Historical Vocabulary Score: {analysis.historical_vocabulary_score:.1f}/10

        Provide specific feedback on historical reasoning, use of evidence, and understanding of causation.
        """

        if source_text:
            base_prompt += f"\n\nReference Material:\n{source_text}\n\nAlso evaluate how well the assignment uses and analyzes the source material."

        return base_prompt

    def _calculate_history_scores(self, analysis: HistoryAnalysis) -> Dict[str, float]:
        """Calculate specialized history scores."""
        scores = {}

        # Historical Accuracy (estimated based on vocabulary and structure)
        scores['historical_accuracy'] = min(analysis.historical_vocabulary_score + 1.0, 10.0)

        # Chronological Understanding
        scores['chronological_understanding'] = analysis.chronological_accuracy

        # Source Analysis
        scores['source_analysis'] = analysis.evidence_usage_score

        # Contextual Awareness
        scores['contextual_awareness'] = analysis.historical_context_score

        # Argument Development
        scores['argument_development'] = analysis.argument_structure_score

        return scores

    def _generate_history_insights(self, analysis: HistoryAnalysis) -> List[str]:
        """Generate insights specific to history assignments."""
        insights = []

        insights.append(f"Focuses on {analysis.period.value.replace('_', ' ')} period")
        insights.append(f"Assignment type: {analysis.assignment_type.value.replace('_', ' ')}")
        insights.append(f"Regional focus: {analysis.region_focus.value.replace('_', ' ')}")

        if analysis.dates_identified:
            insights.append(f"References {len(analysis.dates_identified)} specific dates/time periods")

        if analysis.historical_figures:
            insights.append(f"Mentions {len(analysis.historical_figures)} historical figures")

        if analysis.events_mentioned:
            insights.append(f"Discusses {len(analysis.events_mentioned)} historical events")

        if analysis.sources_cited:
            insights.append(f"Cites {len(analysis.sources_cited)} sources")
        else:
            insights.append("No sources cited - consider adding primary/secondary sources")

        # Bias awareness
        if analysis.bias_awareness_score > 7.0:
            insights.append("Shows good awareness of historical perspective and bias")
        elif analysis.bias_awareness_score < 5.0:
            insights.append("Limited awareness of historical perspective and potential bias")

        return insights

    def _generate_history_recommendations(self, analysis: HistoryAnalysis, scores: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving history assignments."""
        recommendations = []

        # Chronological understanding
        if scores.get('chronological_understanding', 0) < 6.0:
            if len(analysis.dates_identified) == 0:
                recommendations.append("Include specific dates and time periods to establish chronology")
            else:
                recommendations.append("Improve chronological organization and sequencing")

        # Source analysis
        if scores.get('source_analysis', 0) < 6.0:
            if len(analysis.sources_cited) == 0:
                recommendations.append("Include primary and secondary sources to support arguments")
            else:
                recommendations.append("Analyze sources more critically for bias and reliability")

        # Contextual awareness
        if scores.get('contextual_awareness', 0) < 6.0:
            recommendations.append(f"Provide more context about {analysis.period.value.replace('_', ' ')} period conditions")

        # Argument development
        if scores.get('argument_development', 0) < 6.0:
            recommendations.append("Strengthen argument structure with clear thesis and supporting evidence")

        # Historical vocabulary
        if analysis.historical_vocabulary_score < 6.0:
            recommendations.append("Use more historical terminology and concepts")

        # Bias awareness
        if analysis.bias_awareness_score < 5.0:
            recommendations.append("Consider multiple perspectives and potential bias in sources")

        # Period-specific recommendations
        if analysis.period == HistoryPeriod.UNKNOWN:
            recommendations.append("Clearly establish the historical time period being discussed")

        # Assignment type-specific recommendations
        if analysis.assignment_type == HistoryAssignmentType.CAUSE_AND_EFFECT:
            recommendations.append("Clearly explain causal relationships between events")
        elif analysis.assignment_type == HistoryAssignmentType.COMPARATIVE_STUDY:
            recommendations.append("Include both similarities and differences in comparisons")

        return recommendations

def create_history_processor() -> HistoryProcessor:
    """Factory function to create a HistoryProcessor instance."""
    return HistoryProcessor()
