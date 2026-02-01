"""
Punctuation pattern analysis for stylometric profiling.

Punctuation usage is often subconscious and provides strong authorship signals.
"""

import re
from collections import Counter
from typing import Dict, List, Any

from seshat.utils import tokenize_words, tokenize_sentences, safe_divide


class PunctuationFeatures:
    """Extract punctuation features from text."""

    def __init__(self):
        self.punctuation_marks = {
            "period": ".",
            "comma": ",",
            "semicolon": ";",
            "colon": ":",
            "question_mark": "?",
            "exclamation_mark": "!",
            "single_quote": "'",
            "double_quote": '"',
            "apostrophe": "'",
            "hyphen": "-",
            "en_dash": "–",
            "em_dash": "—",
            "left_paren": "(",
            "right_paren": ")",
            "left_bracket": "[",
            "right_bracket": "]",
            "left_brace": "{",
            "right_brace": "}",
            "slash": "/",
            "backslash": "\\",
            "ampersand": "&",
            "at_sign": "@",
            "hash": "#",
            "asterisk": "*",
            "underscore": "_",
            "plus": "+",
            "equals": "=",
            "tilde": "~",
            "backtick": "`",
            "caret": "^",
            "pipe": "|",
        }

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all punctuation features from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of punctuation features
        """
        if not text:
            return self._empty_features()

        words = tokenize_words(text)
        sentences = tokenize_sentences(text)
        total_words = len(words) if words else 1

        features = {}

        frequency_features = self._extract_frequency_features(text, total_words)
        features.update(frequency_features)

        pattern_features = self._extract_pattern_features(text)
        features.update(pattern_features)

        sentence_features = self._extract_sentence_punctuation(text, sentences)
        features.update(sentence_features)

        style_features = self._extract_style_features(text)
        features.update(style_features)

        return features

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary."""
        return {
            "period_per_1k": 0.0,
            "comma_per_1k": 0.0,
            "semicolon_per_1k": 0.0,
            "colon_per_1k": 0.0,
            "question_mark_per_1k": 0.0,
            "exclamation_mark_per_1k": 0.0,
            "quote_per_1k": 0.0,
            "hyphen_per_1k": 0.0,
            "dash_per_1k": 0.0,
            "parenthesis_per_1k": 0.0,
            "ellipsis_per_1k": 0.0,
            "multiple_exclamation_count": 0,
            "multiple_question_count": 0,
            "mixed_punctuation_count": 0,
            "ellipsis_count": 0,
            "comma_per_sentence": 0.0,
            "terminal_period_ratio": 0.0,
            "terminal_question_ratio": 0.0,
            "terminal_exclamation_ratio": 0.0,
            "em_dash_preference": 0.0,
            "single_vs_double_quote": 0.0,
            "oxford_comma_likely": False,
            "punctuation_diversity": 0.0,
        }

    def _extract_frequency_features(
        self, text: str, total_words: int
    ) -> Dict[str, float]:
        """Extract punctuation frequency features (per 1000 words)."""
        multiplier = 1000 / total_words if total_words > 0 else 0

        period_count = text.count(".")
        ellipsis_count = len(re.findall(r"\.{3,}", text))
        adjusted_periods = period_count - (ellipsis_count * 3)

        features = {
            "period_per_1k": adjusted_periods * multiplier,
            "comma_per_1k": text.count(",") * multiplier,
            "semicolon_per_1k": text.count(";") * multiplier,
            "colon_per_1k": text.count(":") * multiplier,
            "question_mark_per_1k": text.count("?") * multiplier,
            "exclamation_mark_per_1k": text.count("!") * multiplier,
            "quote_per_1k": (text.count('"') + text.count("'") + text.count(""") + text.count(""") + text.count("'") + text.count("'")) * multiplier,
            "hyphen_per_1k": text.count("-") * multiplier,
            "dash_per_1k": (text.count("–") + text.count("—")) * multiplier,
            "parenthesis_per_1k": (text.count("(") + text.count(")")) * multiplier,
            "ellipsis_per_1k": ellipsis_count * multiplier,
        }

        return features

    def _extract_pattern_features(self, text: str) -> Dict[str, Any]:
        """Extract punctuation pattern features."""
        multiple_exclamation = re.findall(r"!{2,}", text)
        multiple_question = re.findall(r"\?{2,}", text)
        mixed_punctuation = re.findall(r"[!?]{2,}", text)
        ellipsis = re.findall(r"\.{3,}", text)

        return {
            "multiple_exclamation_count": len(multiple_exclamation),
            "multiple_question_count": len(multiple_question),
            "mixed_punctuation_count": len(mixed_punctuation),
            "ellipsis_count": len(ellipsis),
        }

    def _extract_sentence_punctuation(
        self, text: str, sentences: List[str]
    ) -> Dict[str, float]:
        """Extract per-sentence punctuation features."""
        if not sentences:
            return {
                "comma_per_sentence": 0.0,
                "terminal_period_ratio": 0.0,
                "terminal_question_ratio": 0.0,
                "terminal_exclamation_ratio": 0.0,
            }

        total_sentences = len(sentences)
        total_commas = text.count(",")

        terminal_period = 0
        terminal_question = 0
        terminal_exclamation = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                last_char = sentence[-1]
                if last_char == ".":
                    terminal_period += 1
                elif last_char == "?":
                    terminal_question += 1
                elif last_char == "!":
                    terminal_exclamation += 1

        return {
            "comma_per_sentence": total_commas / total_sentences,
            "terminal_period_ratio": terminal_period / total_sentences,
            "terminal_question_ratio": terminal_question / total_sentences,
            "terminal_exclamation_ratio": terminal_exclamation / total_sentences,
        }

    def _extract_style_features(self, text: str) -> Dict[str, Any]:
        """Extract stylistic punctuation preferences."""
        em_dashes = text.count("—")
        en_dashes = text.count("–")
        hyphens_as_dashes = len(re.findall(r"\s-\s|--", text))
        total_dashes = em_dashes + en_dashes + hyphens_as_dashes

        em_dash_preference = safe_divide(em_dashes, total_dashes) if total_dashes > 0 else 0.5

        single_quotes = text.count("'") + text.count("'") + text.count("'")
        double_quotes = text.count('"') + text.count(""") + text.count(""")
        total_quotes = single_quotes + double_quotes

        single_vs_double = safe_divide(single_quotes, total_quotes) if total_quotes > 0 else 0.5

        oxford_comma_pattern = re.findall(r"\w+,\s+\w+,\s+and\s+\w+", text, re.IGNORECASE)
        no_oxford_pattern = re.findall(r"\w+,\s+\w+\s+and\s+\w+", text, re.IGNORECASE)
        oxford_comma_likely = len(oxford_comma_pattern) > len(no_oxford_pattern)

        punctuation_types_used = set()
        for char in text:
            if char in ".,:;?!\"'-–—()[]{}":
                punctuation_types_used.add(char)
        punctuation_diversity = len(punctuation_types_used) / 15

        return {
            "em_dash_preference": em_dash_preference,
            "single_vs_double_quote": single_vs_double,
            "oxford_comma_likely": oxford_comma_likely,
            "punctuation_diversity": punctuation_diversity,
        }

    def get_punctuation_vector(self, text: str) -> List[float]:
        """
        Get a fixed-length vector of punctuation features.

        Useful for ML models.
        """
        features = self.extract(text)

        vector_keys = [
            "period_per_1k", "comma_per_1k", "semicolon_per_1k", "colon_per_1k",
            "question_mark_per_1k", "exclamation_mark_per_1k", "quote_per_1k",
            "hyphen_per_1k", "dash_per_1k", "parenthesis_per_1k", "ellipsis_per_1k",
            "comma_per_sentence", "terminal_period_ratio", "terminal_question_ratio",
            "terminal_exclamation_ratio", "em_dash_preference", "single_vs_double_quote",
            "punctuation_diversity",
        ]

        return [features.get(key, 0.0) for key in vector_keys]

    def analyze_punctuation_patterns(self, text: str) -> Dict[str, Any]:
        """
        Provide detailed analysis of punctuation patterns.

        Returns human-readable analysis of distinctive patterns.
        """
        features = self.extract(text)

        analysis = {
            "distinctive_patterns": [],
            "punctuation_style": "neutral",
            "formality_indicator": "medium",
        }

        if features["exclamation_mark_per_1k"] > 20:
            analysis["distinctive_patterns"].append("High exclamation mark usage (emphatic style)")
            analysis["punctuation_style"] = "emphatic"

        if features["question_mark_per_1k"] > 15:
            analysis["distinctive_patterns"].append("High question frequency (interrogative style)")

        if features["semicolon_per_1k"] > 5:
            analysis["distinctive_patterns"].append("Semicolon user (formal/academic style)")
            analysis["formality_indicator"] = "high"

        if features["ellipsis_count"] > 3:
            analysis["distinctive_patterns"].append("Frequent ellipsis usage (trailing thoughts)")

        if features["multiple_exclamation_count"] > 0:
            analysis["distinctive_patterns"].append("Uses multiple exclamation marks (!!)")
            analysis["formality_indicator"] = "low"

        if features["em_dash_preference"] > 0.7:
            analysis["distinctive_patterns"].append("Prefers em-dashes for interruptions")
        elif features["em_dash_preference"] < 0.3:
            analysis["distinctive_patterns"].append("Uses hyphens/en-dashes for interruptions")

        if features["oxford_comma_likely"]:
            analysis["distinctive_patterns"].append("Uses Oxford comma")

        return analysis


# Alias for backward compatibility
PunctuationFeatureExtractor = PunctuationFeatures
