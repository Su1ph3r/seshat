"""
Native Language Identification (NLI).

Identifies an author's native language based on L1 interference
patterns in their L2 English writing.
"""

from typing import Dict, List, Any, Optional
from collections import Counter
import re

from seshat.utils import tokenize_words, tokenize_sentences
from seshat.analyzer import Analyzer


class NativeLanguageIdentifier:
    """
    Identify native language from L2 English writing.

    Uses character n-grams, grammatical patterns, and L1 interference
    markers to identify the author's native language.
    """

    def __init__(self):
        """Initialize NLI detector."""
        self.l1_patterns = self._load_l1_patterns()

    def _load_l1_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load patterns characteristic of different L1 backgrounds."""
        return {
            "spanish": {
                "article_errors": ["the life", "the people", "the nature", "the society"],
                "preposition_errors": ["in the morning", "arrive to", "depend of", "consist in"],
                "word_order": ["very much like", "not nothing"],
                "false_friends": ["actually", "realize", "library", "assist"],
                "char_patterns": ["ción", "mente", "able", "ible"],
                "spelling_patterns": ["que", "porque", "pero"],
            },
            "chinese": {
                "article_errors": ["go to school", "in hospital", "at university"],
                "preposition_errors": ["arrive at", "good at", "interested in"],
                "word_order": ["although but", "because so"],
                "plural_errors": ["informations", "advices", "furnitures"],
                "tense_markers": ["yesterday go", "tomorrow will"],
                "char_patterns": [],
            },
            "german": {
                "word_order": ["i also think", "can i help"],
                "preposition_errors": ["wait on", "consist from"],
                "false_friends": ["become", "actual", "chef", "gift"],
                "compound_tendency": True,
                "char_patterns": ["ung", "keit", "lich"],
            },
            "french": {
                "article_errors": ["the france", "the love"],
                "preposition_errors": ["depend of", "interested by"],
                "false_friends": ["actually", "eventually", "assist", "demand"],
                "gender_markers": [],
                "char_patterns": ["ment", "tion", "eur", "eux"],
            },
            "japanese": {
                "article_errors": ["go to school", "at university"],
                "word_order": ["very much"],
                "counter_patterns": [],
                "politeness_markers": [],
                "char_patterns": [],
                "l_r_confusion": True,
            },
            "russian": {
                "article_errors": ["go to university", "in hospital"],
                "preposition_errors": ["depends from", "consist of"],
                "aspect_markers": [],
                "char_patterns": [],
            },
            "arabic": {
                "article_errors": ["the my friend", "the this"],
                "word_order": ["adjective noun"],
                "plural_patterns": [],
                "char_patterns": [],
            },
            "portuguese": {
                "preposition_errors": ["depend of", "arrive to"],
                "false_friends": ["actually", "pretend", "push"],
                "char_patterns": ["ção", "mente", "oso"],
            },
            "korean": {
                "article_errors": ["go to school", "at company"],
                "word_order": ["object before verb patterns"],
                "char_patterns": [],
                "honorific_patterns": [],
            },
            "italian": {
                "preposition_errors": ["depend from", "married with"],
                "false_friends": ["actually", "eventually"],
                "char_patterns": ["zione", "mente", "bile"],
            },
        }

    def identify(self, text: str) -> Dict[str, Any]:
        """
        Identify the likely native language of the author.

        Args:
            text: English text to analyze

        Returns:
            NLI results with language probabilities
        """
        if not text or len(text) < 100:
            return {
                "identified_l1": "unknown",
                "confidence": "low",
                "probabilities": {},
                "indicators": [],
            }

        words = tokenize_words(text)
        sentences = tokenize_sentences(text)
        text_lower = text.lower()

        scores = {}
        indicators = {}

        for language, patterns in self.l1_patterns.items():
            score, found_indicators = self._score_language(
                text_lower, words, sentences, patterns
            )
            scores[language] = score
            if found_indicators:
                indicators[language] = found_indicators

        total_score = sum(scores.values())
        if total_score > 0:
            probabilities = {lang: score / total_score for lang, score in scores.items()}
        else:
            probabilities = {lang: 1 / len(scores) for lang in scores}

        sorted_languages = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        top_language = sorted_languages[0][0]
        top_probability = sorted_languages[0][1]

        if top_probability > 0.5:
            confidence = "high"
        elif top_probability > 0.3:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "identified_l1": top_language,
            "confidence": confidence,
            "probabilities": probabilities,
            "top_candidates": sorted_languages[:5],
            "indicators": indicators,
            "analysis": {
                "article_patterns": self._analyze_articles(text_lower, words),
                "preposition_patterns": self._analyze_prepositions(text_lower),
                "word_order_patterns": self._analyze_word_order(sentences),
            },
        }

    def _score_language(
        self,
        text_lower: str,
        words: List[str],
        sentences: List[str],
        patterns: Dict[str, Any],
    ) -> tuple:
        """Score text against a language's L1 interference patterns."""
        score = 0.0
        found_indicators = []

        if "article_errors" in patterns:
            for error_pattern in patterns["article_errors"]:
                if error_pattern in text_lower:
                    score += 1
                    found_indicators.append(f"Article pattern: '{error_pattern}'")

        if "preposition_errors" in patterns:
            for error_pattern in patterns["preposition_errors"]:
                if error_pattern in text_lower:
                    score += 1
                    found_indicators.append(f"Preposition pattern: '{error_pattern}'")

        if "false_friends" in patterns:
            word_set = set(words)
            for ff in patterns["false_friends"]:
                if ff in word_set:
                    score += 0.3

        if "char_patterns" in patterns:
            for char_pattern in patterns["char_patterns"]:
                count = text_lower.count(char_pattern)
                if count > 0:
                    score += count * 0.1

        return score, found_indicators

    def _analyze_articles(
        self, text_lower: str, words: List[str]
    ) -> Dict[str, Any]:
        """Analyze article usage patterns."""
        article_count = sum(1 for w in words if w in ["a", "an", "the"])
        word_count = len(words)

        article_ratio = article_count / word_count if word_count > 0 else 0

        missing_article_patterns = [
            r"\b(go to) (school|university|hospital|work)\b",
            r"\b(at) (university|school|home)\b",
        ]

        missing_count = 0
        for pattern in missing_article_patterns:
            missing_count += len(re.findall(pattern, text_lower))

        return {
            "article_ratio": article_ratio,
            "possible_omissions": missing_count,
            "style": "article-light" if article_ratio < 0.05 else "normal",
        }

    def _analyze_prepositions(self, text_lower: str) -> Dict[str, Any]:
        """Analyze preposition usage patterns."""
        non_native_patterns = [
            ("depend of", "depend on"),
            ("arrive to", "arrive at"),
            ("consist in", "consist of"),
            ("married with", "married to"),
        ]

        errors_found = []
        for wrong, correct in non_native_patterns:
            if wrong in text_lower:
                errors_found.append({
                    "found": wrong,
                    "expected": correct,
                })

        return {
            "errors_found": errors_found,
            "error_count": len(errors_found),
        }

    def _analyze_word_order(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze word order patterns."""
        unusual_patterns = []

        for sentence in sentences:
            sentence_lower = sentence.lower()

            if re.search(r"although.*but", sentence_lower):
                unusual_patterns.append("although...but (Chinese pattern)")

            if re.search(r"because.*so", sentence_lower):
                unusual_patterns.append("because...so (Chinese pattern)")

        return {
            "unusual_patterns": unusual_patterns,
            "pattern_count": len(unusual_patterns),
        }

    def get_language_profile(self, text: str) -> Dict[str, float]:
        """
        Get a simplified language profile for comparison.

        Returns probability scores for each language.
        """
        result = self.identify(text)
        return result.get("probabilities", {})
