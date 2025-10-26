"""
Spanish Assignment Processor
Handles Spanish language assignments including grammar, vocabulary, comprehension, and culture.
"""

import json
import re
from collections import Counter
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import spacy


class SpanishAssignmentType(Enum):
    """Types of Spanish assignments that can be identified."""

    GRAMMAR = "grammar"
    VOCABULARY = "vocabulary"
    READING_COMPREHENSION = "reading_comprehension"
    WRITING = "writing"
    TRANSLATION = "translation"
    CULTURE = "culture"
    CONVERSATION = "conversation"
    CONJUGATION = "conjugation"
    LISTENING = "listening"
    UNKNOWN = "unknown"


class SpanishGrammarRule:
    """Spanish grammar rule for checking."""

    def __init__(
        self, rule_name: str, pattern: str, description: str, examples: List[str]
    ):
        self.rule_name = rule_name
        self.pattern = pattern
        self.description = description
        self.examples = examples


class SpanishAnalysis:
    """Container for Spanish assignment analysis results."""

    def __init__(self):
        self.assignment_type = SpanishAssignmentType.UNKNOWN
        self.grammar_errors = []
        self.vocabulary_level = "beginner"
        self.verb_conjugations = {}
        self.cultural_references = []
        self.comprehension_questions = []
        self.translation_accuracy = 0.0
        self.fluency_score = 0.0
        self.complexity_score = 0.0

    def to_dict(self):
        """Convert analysis to dictionary for serialization."""
        return {
            "assignment_type": (
                self.assignment_type.value
                if hasattr(self.assignment_type, "value")
                else str(self.assignment_type)
            ),
            "grammar_errors": self.grammar_errors,
            "vocabulary_level": self.vocabulary_level,
            "verb_conjugations": self.verb_conjugations,
            "cultural_references": self.cultural_references,
            "comprehension_questions": self.comprehension_questions,
            "translation_accuracy": self.translation_accuracy,
            "fluency_score": self.fluency_score,
            "complexity_score": self.complexity_score,
        }


class SpanishProcessor:
    """Comprehensive Spanish assignment processor."""

    def __init__(self, llm_manager: Any = None):
        # Try to load Spanish spaCy model
        try:
            self.nlp = spacy.load("es_core_news_sm")
            self.spacy_available = True
        except OSError:
            print(
                "[WARNING] Spanish spaCy model not available. Install with: python -m spacy download es_core_news_sm"
            )
            self.spacy_available = False
            self.nlp = None

        self.llm_manager = llm_manager
        self.spanish_vocabulary = self._load_spanish_vocabulary()
        self.grammar_rules = self._load_grammar_rules()
        self.verb_conjugations = self._load_verb_conjugations()
        self.cultural_keywords = self._load_cultural_keywords()

        # Assignment type patterns
        self.assignment_patterns = {
            SpanishAssignmentType.GRAMMAR: [
                r"completa.*oración|complete.*sentence",
                r"conjuga.*verbo|conjugate.*verb",
                r"artículo|article.*el|la|los|las",
                r"adjetivo|adjective",
                r"sustantivo|noun",
                r"tiempo.*verbal|verb.*tense",
            ],
            SpanishAssignmentType.VOCABULARY: [
                r"define.*palabra|define.*word",
                r"vocabulario|vocabulary",
                r"significado|meaning",
                r"sinónimo|synonym",
                r"antónimo|antonym",
                r"palabra.*día|word.*day",
            ],
            SpanishAssignmentType.READING_COMPREHENSION: [
                r"lee.*texto|read.*text",
                r"comprensión|comprehension",
                r"responde.*pregunta|answer.*question",
                r"según.*texto|according.*text",
                r"¿qué|¿cuál|¿cómo|¿dónde|¿cuándo",
            ],
            SpanishAssignmentType.TRANSLATION: [
                r"traduce|translate",
                r"inglés.*español|english.*spanish",
                r"español.*inglés|spanish.*english",
                r"en.*español|in.*spanish",
                r"en.*inglés|in.*english",
            ],
            SpanishAssignmentType.CULTURE: [
                r"cultura|culture",
                r"tradición|tradition",
                r"país|country.*hispanic",
                r"fiesta|celebration",
                r"costumbre|custom",
                r"historia.*españa|history.*spain",
            ],
            SpanishAssignmentType.CONJUGATION: [
                r"presente|present.*tense",
                r"pretérito|preterite",
                r"imperfecto|imperfect",
                r"futuro|future.*tense",
                r"subjuntivo|subjunctive",
                r"imperativo|imperative",
            ],
        }

    def _load_spanish_vocabulary(self) -> Dict[str, List[str]]:
        """Load Spanish vocabulary by level."""
        return {
            "beginner": [
                "hola",
                "adiós",
                "gracias",
                "por favor",
                "sí",
                "no",
                "agua",
                "comida",
                "casa",
                "escuela",
                "familia",
                "amigo",
                "gato",
                "perro",
                "rojo",
                "azul",
                "grande",
                "pequeño",
                "bueno",
                "malo",
                "nuevo",
                "viejo",
            ],
            "intermediate": [
                "aunque",
                "mientras",
                "durante",
                "después",
                "antes",
                "sin embargo",
                "además",
                "por tanto",
                "desarrollo",
                "experiencia",
                "importante",
                "necesario",
                "posible",
                "diferente",
                "especial",
                "internacional",
            ],
            "advanced": [
                "consecuentemente",
                "no obstante",
                "por consiguiente",
                "asimismo",
                "establecimiento",
                "procedimiento",
                "acontecimiento",
                "desenvolvimiento",
                "perfeccionamiento",
                "incomprensible",
                "extraordinario",
                "indispensable",
            ],
        }

    def _load_grammar_rules(self) -> List[SpanishGrammarRule]:
        """Load Spanish grammar rules for checking."""
        return [
            SpanishGrammarRule(
                "article_agreement",
                r"(el|la|los|las)\s+(.*?)(?=\s|$)",
                "Articles must agree with noun gender and number",
                ["el libro", "la mesa", "los niños", "las casas"],
            ),
            SpanishGrammarRule(
                "adjective_agreement",
                r"(.*?)\s+(alto|alta|altos|altas|bueno|buena|buenos|buenas)",
                "Adjectives must agree with noun gender and number",
                ["niño alto", "niña alta", "libros buenos", "casas buenas"],
            ),
            SpanishGrammarRule(
                "ser_estar_usage",
                r"\b(es|está|son|están)\s+",
                "Correct usage of ser vs estar",
                ["Él es médico", "Ella está cansada"],
            ),
        ]

    def _load_verb_conjugations(self) -> Dict[str, Dict[str, List[str]]]:
        """Load common Spanish verb conjugations."""
        return {
            "hablar": {
                "presente": [
                    "hablo",
                    "hablas",
                    "habla",
                    "hablamos",
                    "habláis",
                    "hablan",
                ],
                "pretérito": [
                    "hablé",
                    "hablaste",
                    "habló",
                    "hablamos",
                    "hablasteis",
                    "hablaron",
                ],
            },
            "comer": {
                "presente": ["como", "comes", "come", "comemos", "coméis", "comen"],
                "pretérito": [
                    "comí",
                    "comiste",
                    "comió",
                    "comimos",
                    "comisteis",
                    "comieron",
                ],
            },
            "vivir": {
                "presente": ["vivo", "vives", "vive", "vivimos", "vivís", "viven"],
                "pretérito": [
                    "viví",
                    "viviste",
                    "vivió",
                    "vivimos",
                    "vivisteis",
                    "vivieron",
                ],
            },
        }

    def _load_cultural_keywords(self) -> List[str]:
        """Load Spanish cultural keywords."""
        return [
            "flamenco",
            "tapas",
            "siesta",
            "paella",
            "quinceañera",
            "día de los muertos",
            "semana santa",
            "corrida de toros",
            "sangría",
            "gazpacho",
            "churros",
            "mariachi",
            "salsa",
            "tango",
            "cumbia",
            "reggaeton",
            "españa",
            "méxico",
            "argentina",
            "colombia",
            "perú",
            "chile",
            "venezuela",
            "ecuador",
        ]

    def identify_assignment_type(self, text: str) -> SpanishAssignmentType:
        """Identify the type of Spanish assignment."""
        text_lower = text.lower()

        # Count matches for each assignment type
        type_scores = {}
        for assignment_type, patterns in self.assignment_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            type_scores[assignment_type] = score

        # Return the type with the highest score
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            return best_type[0] if best_type[1] > 0 else SpanishAssignmentType.UNKNOWN

        return SpanishAssignmentType.UNKNOWN

    def analyze_vocabulary_level(self, text: str) -> str:
        """Analyze the vocabulary level of Spanish text."""
        words = re.findall(r"\b[a-záéíóúñü]+\b", text.lower())

        beginner_count = sum(
            1 for word in words if word in self.spanish_vocabulary["beginner"]
        )
        intermediate_count = sum(
            1 for word in words if word in self.spanish_vocabulary["intermediate"]
        )
        advanced_count = sum(
            1 for word in words if word in self.spanish_vocabulary["advanced"]
        )

        total_known = beginner_count + intermediate_count + advanced_count

        if total_known == 0:
            return "unknown"

        advanced_ratio = advanced_count / total_known
        intermediate_ratio = intermediate_count / total_known

        if advanced_ratio > 0.3:
            return "advanced"
        elif intermediate_ratio > 0.4:
            return "intermediate"
        else:
            return "beginner"

    def check_grammar(self, text: str) -> List[Dict[str, Any]]:
        """Check Spanish grammar in the text."""
        errors = []

        for rule in self.grammar_rules:
            matches = re.finditer(rule.pattern, text, re.IGNORECASE)
            for match in matches:
                # Basic grammar checking logic
                matched_text = match.group()

                if rule.rule_name == "article_agreement":
                    article = match.group(1).lower()
                    noun_phrase = match.group(2).lower()

                    # Check for obvious mismatches
                    if (
                        article in ["el", "los"]
                        and any(fem in noun_phrase for fem in ["a ", "as "])
                    ) or (
                        article in ["la", "las"]
                        and any(masc in noun_phrase for masc in ["o ", "os "])
                    ):
                        errors.append(
                            {
                                "rule": rule.rule_name,
                                "text": matched_text,
                                "description": rule.description,
                                "position": match.start(),
                            }
                        )

        return errors

    def analyze_verb_conjugations(self, text: str) -> Dict[str, List[str]]:
        """Analyze verb conjugations in Spanish text."""
        found_conjugations = {}

        for verb, conjugations in self.verb_conjugations.items():
            for tense, forms in conjugations.items():
                found_forms = []
                for form in forms:
                    if re.search(rf"\b{form}\b", text, re.IGNORECASE):
                        found_forms.append(form)

                if found_forms:
                    if verb not in found_conjugations:
                        found_conjugations[verb] = {}
                    found_conjugations[verb][tense] = found_forms

        return found_conjugations

    def extract_cultural_references(self, text: str) -> List[str]:
        """Extract Spanish cultural references from text."""
        found_cultural = []
        text_lower = text.lower()

        for keyword in self.cultural_keywords:
            if keyword in text_lower:
                found_cultural.append(keyword)

        return found_cultural

    def analyze_reading_comprehension(self, text: str) -> List[Dict[str, str]]:
        """Extract and analyze reading comprehension questions."""
        questions = []

        # Spanish question patterns
        question_patterns = [
            r"¿[^?]*\?",  # Questions with Spanish question marks
            r"[A-Z][^.!?]*[?]",  # Questions ending with ?
            r"(?:Qué|Cuál|Cómo|Dónde|Cuándo|Por qué|Quién)[^.!?]*[?]",  # Specific question words
        ]

        for pattern in question_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                question_type = "unknown"
                match_lower = match.lower()

                if any(word in match_lower for word in ["qué", "what"]):
                    question_type = "what"
                elif any(word in match_lower for word in ["cuál", "which"]):
                    question_type = "which"
                elif any(word in match_lower for word in ["cómo", "how"]):
                    question_type = "how"
                elif any(word in match_lower for word in ["dónde", "where"]):
                    question_type = "where"
                elif any(word in match_lower for word in ["cuándo", "when"]):
                    question_type = "when"
                elif any(word in match_lower for word in ["por qué", "why"]):
                    question_type = "why"
                elif any(word in match_lower for word in ["quién", "who"]):
                    question_type = "who"

                questions.append({"question": match.strip(), "type": question_type})

        return questions

    def calculate_fluency_score(self, text: str) -> float:
        """Calculate Spanish fluency score based on various factors."""
        score = 0.0

        # Sentence structure variety (0-25 points)
        sentences = re.split(r"[.!?]+", text)
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(
            len(sentences), 1
        )
        if 8 <= avg_sentence_length <= 20:
            score += 25
        elif 5 <= avg_sentence_length <= 25:
            score += 15
        else:
            score += 5

        # Vocabulary diversity (0-25 points)
        words = re.findall(r"\b[a-záéíóúñü]+\b", text.lower())
        unique_words = len(set(words))
        total_words = len(words)
        diversity_ratio = unique_words / max(total_words, 1)
        score += min(diversity_ratio * 25, 25)

        # Grammar complexity (0-25 points)
        complex_structures = len(
            re.findall(
                r"\b(aunque|mientras|durante|después|sin embargo|por tanto)\b",
                text.lower(),
            )
        )
        score += min(complex_structures * 5, 25)

        # Proper accents and special characters (0-25 points)
        spanish_chars = len(re.findall(r"[áéíóúñü]", text.lower()))
        total_chars = len(re.findall(r"[a-záéíóúñü]", text.lower()))
        if total_chars > 0:
            accent_ratio = spanish_chars / total_chars
            score += min(accent_ratio * 100, 25)

        return min(score, 100.0)

    def analyze_spanish_assignment(self, text: str) -> SpanishAnalysis:
        """Comprehensive analysis of Spanish assignment."""
        analysis = SpanishAnalysis()

        # Identify assignment type
        analysis.assignment_type = self.identify_assignment_type(text)

        # Check grammar
        analysis.grammar_errors = self.check_grammar(text)

        # Analyze vocabulary level
        analysis.vocabulary_level = self.analyze_vocabulary_level(text)

        # Analyze verb conjugations
        analysis.verb_conjugations = self.analyze_verb_conjugations(text)

        # Extract cultural references
        analysis.cultural_references = self.extract_cultural_references(text)

        # Analyze reading comprehension
        analysis.comprehension_questions = self.analyze_reading_comprehension(text)

        # Calculate fluency score
        analysis.fluency_score = self.calculate_fluency_score(text)

        # Calculate complexity score
        words = len(text.split())
        sentences = len(re.split(r"[.!?]+", text))
        avg_words_per_sentence = words / max(sentences, 1)
        analysis.complexity_score = min((avg_words_per_sentence / 15) * 100, 100)

        return analysis

    def grade_spanish_assignment(
        self, assignment_text: str, source_text: str = None
    ) -> Dict[str, Any]:
        """Grade Spanish assignment with language-specific criteria."""
        analysis = self.analyze_spanish_assignment(assignment_text)

        # Calculate grades based on Spanish-specific criteria
        grammar_accuracy = max(10 - len(analysis.grammar_errors), 0)
        vocabulary_usage = self._score_vocabulary_usage(analysis.vocabulary_level)
        fluency_communication = analysis.fluency_score / 10
        cultural_understanding = min(len(analysis.cultural_references) * 2, 10)

        # Adjust scores based on assignment type
        if analysis.assignment_type == SpanishAssignmentType.GRAMMAR:
            grammar_accuracy *= 1.5  # Grammar more important
            fluency_communication *= 0.8
        elif analysis.assignment_type == SpanishAssignmentType.VOCABULARY:
            vocabulary_usage *= 1.5  # Vocabulary more important
            grammar_accuracy *= 0.8
        elif analysis.assignment_type == SpanishAssignmentType.CULTURE:
            cultural_understanding *= 1.5  # Culture more important
            grammar_accuracy *= 0.9

        # Normalize scores to 0-10
        grammar_accuracy = min(grammar_accuracy, 10)
        vocabulary_usage = min(vocabulary_usage, 10)
        fluency_communication = min(fluency_communication, 10)
        cultural_understanding = min(cultural_understanding, 10)

        return {
            "grammar_accuracy": grammar_accuracy,
            "vocabulary_usage": vocabulary_usage,
            "fluency_communication": fluency_communication,
            "cultural_understanding": cultural_understanding,
            "overall_score": (
                grammar_accuracy * 0.3
                + vocabulary_usage * 0.25
                + fluency_communication * 0.3
                + cultural_understanding * 0.15
            ),
            "analysis": analysis,
            "feedback": self._generate_spanish_feedback(analysis),
        }

    def _score_vocabulary_usage(self, level: str) -> float:
        """Score vocabulary usage based on level."""
        level_scores = {
            "beginner": 6.0,
            "intermediate": 8.0,
            "advanced": 10.0,
            "unknown": 3.0,
        }
        return level_scores.get(level, 3.0)

    def _generate_spanish_feedback(self, analysis: SpanishAnalysis) -> List[str]:
        """Generate specific feedback for Spanish assignments."""
        feedback = []

        if len(analysis.grammar_errors) > 3:
            feedback.append(
                "Revisa la concordancia entre artículos, sustantivos y adjetivos."
            )

        if analysis.vocabulary_level == "beginner":
            feedback.append("Intenta usar vocabulario más variado y avanzado.")

        if analysis.fluency_score < 50:
            feedback.append("Trabaja en crear oraciones más complejas y naturales.")

        if (
            len(analysis.cultural_references) == 0
            and analysis.assignment_type == SpanishAssignmentType.CULTURE
        ):
            feedback.append(
                "Incluye más referencias culturales específicas del mundo hispanohablante."
            )

        if len(analysis.verb_conjugations) < 2:
            feedback.append("Usa una mayor variedad de tiempos verbales.")

        if not feedback:
            feedback.append(
                "¡Excelente trabajo en español! Tu uso del idioma es claro y apropiado."
            )

        return feedback


def create_spanish_processor() -> SpanishProcessor:
    """Factory function to create a Spanish processor instance."""
    return SpanishProcessor()
