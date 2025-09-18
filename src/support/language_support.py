"""
Multi-language support module for assignment grading system.
Provides language detection, localization, and language-specific processing capabilities.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)

class SupportedLanguage(Enum):
    """Supported languages with their ISO codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"

@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    primary_language: str
    confidence: float
    all_detected: List[Tuple[str, float]]
    is_supported: bool
    fallback_language: str = "en"

@dataclass
class LanguageConfig:
    """Configuration for a specific language."""
    code: str
    name: str
    tesseract_lang: str
    grammar_tool_code: str
    right_to_left: bool = False
    requires_special_handling: bool = False

class LanguageManager:
    """Manages multi-language support for the assignment grading system."""

    def __init__(self):
        """Initialize language manager."""
        self.supported_languages = self._initialize_language_configs()
        self.default_language = SupportedLanguage.ENGLISH.value
        self.prompts = self._load_prompts()

    def _initialize_language_configs(self) -> Dict[str, LanguageConfig]:
        """Initialize language configurations."""
        configs = {
            "en": LanguageConfig("en", "English", "eng", "en-US"),
            "es": LanguageConfig("es", "Spanish", "spa", "es"),
            "fr": LanguageConfig("fr", "French", "fra", "fr"),
            "de": LanguageConfig("de", "German", "deu", "de-DE"),
            "it": LanguageConfig("it", "Italian", "ita", "it"),
            "pt": LanguageConfig("pt", "Portuguese", "por", "pt-PT"),
            "nl": LanguageConfig("nl", "Dutch", "nld", "nl"),
            "ru": LanguageConfig("ru", "Russian", "rus", "ru", requires_special_handling=True),
            "zh-cn": LanguageConfig("zh-cn", "Chinese (Simplified)", "chi_sim", "zh-CN", requires_special_handling=True),
            "zh-tw": LanguageConfig("zh-tw", "Chinese (Traditional)", "chi_tra", "zh-TW", requires_special_handling=True),
            "ja": LanguageConfig("ja", "Japanese", "jpn", "ja", requires_special_handling=True),
            "ko": LanguageConfig("ko", "Korean", "kor", "ko", requires_special_handling=True),
            "ar": LanguageConfig("ar", "Arabic", "ara", "ar", right_to_left=True, requires_special_handling=True),
            "hi": LanguageConfig("hi", "Hindi", "hin", "hi", requires_special_handling=True),
        }
        return configs

    def _load_prompts(self) -> Dict[str, Dict[str, str]]:
        """Load language-specific prompts."""
        return {
            "en": {
                "grammar_check": "Count grammatical errors in the following English text:\n{text}",
                "plagiarism_check": """Analyze the following English assignment for potential plagiarism.
Identify any phrases or sections that appear copied or unoriginal.
Provide a brief summary of your findings and a plagiarism likelihood score (0–100).

Assignment:
{text}""",
                "relevance_check": """Analyze how well the English assignment relates to the provided source material. Focus only on evaluation, not suggestions for improvement.

Evaluate the following aspects:
1. Does the assignment address the main topics from the source?
2. Are the facts presented consistent with the source material?
3. How closely does the assignment content align with the source?

Provide only an analytical assessment without offering revisions or examples of improvements.

Assignment:
{text}

Source:
{source}""",
                "grading_prompt": """You are an academic evaluator. Grade the following English student assignment based on four criteria:
1. Factual Accuracy (0–10): How accurate is the content compared to the source?
2. Relevance to Source (0–10): How well does the assignment relate to the source material?
3. Coherence (0–10): How well-structured and logical is the writing?
4. Grammar (1–10): How well is the assignment written in English? Evaluate spelling, grammar, sentence structure, and writing quality.

Assignment:
{answer}

Source Material:
{source}

Return your response as a JSON object like:
{{
  "factuality": float,
  "relevance": float,
  "coherence": float,
  "grammar": float
}}
Only return the JSON. Do not include any explanation or formatting.""",
                "summary_prompt": "IMPORTANT: Write ONLY a 2-3 sentence summary. Do NOT write any introduction, preamble, or phrases like 'Here is a summary' or 'This assignment'. Start your response immediately with the actual content summary.\n\nText to summarize:\n{text}"
            },
            "es": {
                "grammar_check": "Cuenta los errores gramaticales en el siguiente texto en español:\n{text}",
                "plagiarism_check": """Analiza la siguiente tarea en español para detectar posible plagio.
Identifica cualquier frase o sección que parezca copiada o no original.
Proporciona un resumen breve de tus hallazgos y una puntuación de probabilidad de plagio (0–100).

Tarea:
{text}""",
                "relevance_check": """Analiza qué tan bien se relaciona la tarea en español con el material fuente proporcionado. Enfócate solo en la evaluación, no en sugerencias de mejora.

Evalúa los siguientes aspectos:
1. ¿La tarea aborda los temas principales de la fuente?
2. ¿Los hechos presentados son consistentes con el material fuente?
3. ¿Qué tan estrechamente se alinea el contenido de la tarea con la fuente?

Proporciona solo una evaluación analítica sin ofrecer revisiones o ejemplos de mejoras.

Tarea:
{text}

Fuente:
{source}""",
                "grading_prompt": """Eres un evaluador académico. Califica la siguiente tarea de estudiante en español basándote en cuatro criterios:
1. Precisión Factual (0–10): ¿Qué tan preciso es el contenido comparado con la fuente?
2. Relevancia a la Fuente (0–10): ¿Qué tan bien se relaciona la tarea con el material fuente?
3. Coherencia (0–10): ¿Qué tan bien estructurada y lógica es la escritura?
4. Gramática (1–10): ¿Qué tan bien está escrita la tarea en español? Evalúa ortografía, gramática, estructura de oraciones y calidad de escritura.

Tarea:
{answer}

Material Fuente:
{source}

Devuelve tu respuesta como un objeto JSON así:
{{
  "factuality": float,
  "relevance": float,
  "coherence": float,
  "grammar": float
}}
Solo devuelve el JSON. No incluyas explicación o formato.""",
                "summary_prompt": "IMPORTANTE: Escribe SOLO un resumen de 2-3 oraciones. NO escribas introducción, preámbulo o frases como 'Aquí está el resumen'. Comienza inmediatamente con el contenido del resumen.\n\nTexto a resumir:\n{text}"
            },
            "fr": {
                "grammar_check": "Comptez les erreurs grammaticales dans le texte français suivant:\n{text}",
                "plagiarism_check": """Analysez le devoir français suivant pour détecter un plagiat potentiel.
Identifiez toute phrase ou section qui semble copiée ou non originale.
Fournissez un résumé bref de vos découvertes et un score de probabilité de plagiat (0–100).

Devoir:
{text}""",
                "relevance_check": """Analysez dans quelle mesure le devoir français se rapporte au matériel source fourni. Concentrez-vous uniquement sur l'évaluation, pas sur les suggestions d'amélioration.

Évaluez les aspects suivants:
1. Le devoir aborde-t-il les sujets principaux de la source?
2. Les faits présentés sont-ils cohérents avec le matériel source?
3. Dans quelle mesure le contenu du devoir s'aligne-t-il avec la source?

Fournissez uniquement une évaluation analytique sans offrir de révisions ou d'exemples d'améliorations.

Devoir:
{text}

Source:
{source}""",
                "grading_prompt": """Vous êtes un évaluateur académique. Notez le devoir d'étudiant français suivant selon quatre critères:
1. Précision Factuelle (0–10): À quel point le contenu est-il précis comparé à la source?
2. Pertinence à la Source (0–10): Dans quelle mesure le devoir se rapporte-t-il au matériel source?
3. Cohérence (0–10): À quel point l'écriture est-elle bien structurée et logique?
4. Grammaire (1–10): À quel point le devoir est-il bien écrit en français? Évaluez l'orthographe, la grammaire, la structure des phrases et la qualité d'écriture.

Devoir:
{answer}

Matériel Source:
{source}

Retournez votre réponse comme un objet JSON ainsi:
{{
  "factuality": float,
  "relevance": float,
  "coherence": float,
  "grammar": float
}}
Ne retournez que le JSON. N'incluez aucune explication ou formatage.""",
                "summary_prompt": "IMPORTANT: Écrivez SEULEMENT un résumé de 2-3 phrases. N'écrivez PAS d'introduction, de préambule ou de phrases comme 'Voici le résumé'. Commencez immédiatement par le contenu du résumé.\n\nTexte à résumer:\n{text}"
            }
        }

    def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        Detect the language of the given text.

        Args:
            text: Text to analyze

        Returns:
            LanguageDetectionResult with detection details
        """
        if not LANGDETECT_AVAILABLE:
            logger.warning("Language detection not available, defaulting to English")
            return LanguageDetectionResult(
                primary_language=self.default_language,
                confidence=0.5,
                all_detected=[("en", 0.5)],
                is_supported=True,
                fallback_language="en"
            )

        try:
            # Clean text for better detection
            cleaned_text = self._clean_text_for_detection(text)

            if len(cleaned_text.strip()) < 10:
                logger.warning("Text too short for reliable language detection")
                return LanguageDetectionResult(
                    primary_language=self.default_language,
                    confidence=0.3,
                    all_detected=[("en", 0.3)],
                    is_supported=True,
                    fallback_language="en"
                )

            # Detect primary language
            primary_lang = detect(cleaned_text)

            # Get all detected languages with probabilities
            lang_probs = detect_langs(cleaned_text)
            all_detected = [(lang.lang, lang.prob) for lang in lang_probs]

            # Check if primary language is supported
            is_supported = primary_lang in self.supported_languages

            # Use fallback if not supported
            if not is_supported:
                fallback_lang = self._find_best_fallback(primary_lang)
                logger.info(f"Language {primary_lang} not supported, using fallback: {fallback_lang}")
            else:
                fallback_lang = primary_lang

            return LanguageDetectionResult(
                primary_language=primary_lang,
                confidence=lang_probs[0].prob if lang_probs else 0.0,
                all_detected=all_detected,
                is_supported=is_supported,
                fallback_language=fallback_lang
            )

        except LangDetectException as e:
            logger.error(f"Language detection failed: {e}")
            return LanguageDetectionResult(
                primary_language=self.default_language,
                confidence=0.1,
                all_detected=[("en", 0.1)],
                is_supported=True,
                fallback_language="en"
            )

    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text to improve language detection accuracy."""
        import re

        # Remove metadata headers (Name:, Date:, etc.)
        text = re.sub(r'^(Name|Date|Class|Subject|Source):\s*.*$', '', text, flags=re.MULTILINE)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _find_best_fallback(self, detected_lang: str) -> str:
        """Find the best supported fallback language."""
        # Language family mappings for better fallbacks
        fallback_map = {
            'ca': 'es',  # Catalan -> Spanish
            'gl': 'es',  # Galician -> Spanish
            'eu': 'es',  # Basque -> Spanish
            'ro': 'it',  # Romanian -> Italian
            'da': 'de',  # Danish -> German
            'no': 'de',  # Norwegian -> German
            'sv': 'de',  # Swedish -> German
            'fi': 'de',  # Finnish -> German
            'pl': 'de',  # Polish -> German
            'cs': 'de',  # Czech -> German
            'sk': 'de',  # Slovak -> German
            'hu': 'de',  # Hungarian -> German
            'bg': 'ru',  # Bulgarian -> Russian
            'uk': 'ru',  # Ukrainian -> Russian
            'be': 'ru',  # Belarusian -> Russian
            'sr': 'ru',  # Serbian -> Russian
            'hr': 'de',  # Croatian -> German
            'sl': 'de',  # Slovenian -> German
            'th': 'en',  # Thai -> English
            'vi': 'en',  # Vietnamese -> English
            'id': 'en',  # Indonesian -> English
            'ms': 'en',  # Malay -> English
            'tl': 'en',  # Filipino -> English
        }

        return fallback_map.get(detected_lang, self.default_language)

    def get_language_config(self, language_code: str) -> LanguageConfig:
        """Get configuration for a specific language."""
        return self.supported_languages.get(language_code, self.supported_languages[self.default_language])

    def get_prompt(self, prompt_type: str, language_code: str) -> str:
        """
        Get a localized prompt for the specified language.

        Args:
            prompt_type: Type of prompt (grammar_check, plagiarism_check, etc.)
            language_code: Language code

        Returns:
            Localized prompt string
        """
        lang_prompts = self.prompts.get(language_code, self.prompts[self.default_language])
        return lang_prompts.get(prompt_type, self.prompts[self.default_language].get(prompt_type, ""))

    def get_tesseract_languages(self, language_codes: List[str]) -> str:
        """
        Get Tesseract language parameter for OCR.

        Args:
            language_codes: List of language codes

        Returns:
            Tesseract language parameter string
        """
        tesseract_langs = []
        for code in language_codes:
            config = self.get_language_config(code)
            if config.tesseract_lang not in tesseract_langs:
                tesseract_langs.append(config.tesseract_lang)

        return '+'.join(tesseract_langs)

    def get_supported_languages_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all supported languages."""
        return {
            code: {
                "name": config.name,
                "code": config.code,
                "tesseract_support": config.tesseract_lang,
                "grammar_tool_support": config.grammar_tool_code,
                "right_to_left": config.right_to_left,
                "requires_special_handling": config.requires_special_handling
            }
            for code, config in self.supported_languages.items()
        }

    def is_rtl_language(self, language_code: str) -> bool:
        """Check if language is right-to-left."""
        config = self.get_language_config(language_code)
        return config.right_to_left

    def requires_special_handling(self, language_code: str) -> bool:
        """Check if language requires special processing."""
        config = self.get_language_config(language_code)
        return config.requires_special_handling

    def format_prompt(self, prompt_type: str, language_code: str, **kwargs) -> str:
        """
        Format a localized prompt with the provided parameters.

        Args:
            prompt_type: Type of prompt
            language_code: Language code
            **kwargs: Variables to format into the prompt

        Returns:
            Formatted prompt string
        """
        prompt_template = self.get_prompt(prompt_type, language_code)
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing parameter {e} for prompt formatting, using English fallback")
            fallback_prompt = self.get_prompt(prompt_type, self.default_language)
            return fallback_prompt.format(**kwargs)

# Global language manager instance
language_manager = LanguageManager()

def detect_text_language(text: str) -> LanguageDetectionResult:
    """
    Detect the language of text.

    Args:
        text: Text to analyze

    Returns:
        LanguageDetectionResult
    """
    return language_manager.detect_language(text)

def get_localized_prompt(prompt_type: str, language_code: str, **kwargs) -> str:
    """
    Get a localized and formatted prompt.

    Args:
        prompt_type: Type of prompt
        language_code: Language code
        **kwargs: Variables for prompt formatting

    Returns:
        Formatted localized prompt
    """
    return language_manager.format_prompt(prompt_type, language_code, **kwargs)

def get_supported_languages() -> Dict[str, Dict[str, Any]]:
    """Get information about supported languages."""
    return language_manager.get_supported_languages_info()

def install_tesseract_language_pack(language_code: str) -> bool:
    """
    Install Tesseract language pack for the specified language.

    Args:
        language_code: Language code

    Returns:
        True if successful, False otherwise
    """
    try:
        import subprocess
        config = language_manager.get_language_config(language_code)

        # Try to install via brew (macOS)
        result = subprocess.run([
            'brew', 'install', f'tesseract-lang'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Successfully installed Tesseract language pack for {config.name}")
            return True
        else:
            logger.warning(f"Failed to install Tesseract language pack for {config.name}: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error installing Tesseract language pack: {e}")
        return False

if __name__ == "__main__":
    # Test language detection and support
    test_texts = {
        "en": "This is a test assignment written in English about environmental science.",
        "es": "Esta es una tarea de prueba escrita en español sobre ciencias ambientales.",
        "fr": "Ceci est un devoir de test écrit en français sur les sciences de l'environnement.",
        "de": "Dies ist eine Testaufgabe, die auf Deutsch über Umweltwissenschaften geschrieben wurde.",
    }

    print("Multi-Language Support Test Results:")
    print("=" * 50)

    for lang_code, text in test_texts.items():
        result = detect_text_language(text)
        print(f"\nTest for {lang_code}:")
        print(f"  Detected: {result.primary_language} (confidence: {result.confidence:.2f})")
        print(f"  Supported: {result.is_supported}")
        print(f"  Fallback: {result.fallback_language}")

        # Test prompt localization
        prompt = get_localized_prompt("grammar_check", result.fallback_language, text=text[:50] + "...")
        print(f"  Sample prompt: {prompt[:100]}...")

    print(f"\nSupported Languages: {len(get_supported_languages())}")
    for code, info in get_supported_languages().items():
        print(f"  {code}: {info['name']}")