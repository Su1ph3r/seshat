"""
Formatting feature extraction for stylometric analysis.

Includes whitespace patterns, paragraph structure, capitalization habits,
and other formatting characteristics.
"""

import re
from collections import Counter
from typing import Dict, List, Any

from seshat.utils import (
    tokenize_sentences,
    tokenize_paragraphs,
    tokenize_words_preserve_case,
    safe_divide,
)


class FormattingFeatures:
    """Extract formatting features from text."""

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all formatting features from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of formatting features
        """
        if not text:
            return self._empty_features()

        features = {}

        whitespace_features = self._extract_whitespace_features(text)
        features.update(whitespace_features)

        structure_features = self._extract_structure_features(text)
        features.update(structure_features)

        capitalization_features = self._extract_capitalization_features(text)
        features.update(capitalization_features)

        list_features = self._extract_list_features(text)
        features.update(list_features)

        return features

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary."""
        return {
            "multiple_space_count": 0,
            "multiple_space_ratio": 0.0,
            "trailing_whitespace_count": 0,
            "tab_count": 0,
            "tab_ratio": 0.0,
            "line_break_count": 0,
            "double_line_break_count": 0,
            "avg_line_length": 0.0,
            "paragraph_count": 0,
            "avg_paragraph_length_sentences": 0.0,
            "avg_paragraph_length_words": 0.0,
            "single_sentence_paragraph_ratio": 0.0,
            "all_caps_word_count": 0,
            "all_caps_word_ratio": 0.0,
            "title_case_word_count": 0,
            "title_case_word_ratio": 0.0,
            "sentence_initial_cap_consistency": 0.0,
            "mixed_case_word_count": 0,
            "bullet_list_count": 0,
            "numbered_list_count": 0,
            "uses_bullet_lists": False,
            "uses_numbered_lists": False,
            "indentation_spaces_avg": 0.0,
        }

    def _extract_whitespace_features(self, text: str) -> Dict[str, Any]:
        """Extract whitespace-related features."""
        multiple_spaces = re.findall(r" {2,}", text)
        multiple_space_count = len(multiple_spaces)

        total_chars = len(text)
        multiple_space_chars = sum(len(s) for s in multiple_spaces)

        lines = text.split("\n")
        trailing_whitespace = sum(1 for line in lines if line != line.rstrip())

        tab_count = text.count("\t")

        line_break_count = text.count("\n")
        double_line_breaks = len(re.findall(r"\n\s*\n", text))

        line_lengths = [len(line) for line in lines if line.strip()]
        avg_line_length = sum(line_lengths) / len(line_lengths) if line_lengths else 0

        return {
            "multiple_space_count": multiple_space_count,
            "multiple_space_ratio": safe_divide(multiple_space_chars, total_chars),
            "trailing_whitespace_count": trailing_whitespace,
            "tab_count": tab_count,
            "tab_ratio": safe_divide(tab_count, total_chars),
            "line_break_count": line_break_count,
            "double_line_break_count": double_line_breaks,
            "avg_line_length": avg_line_length,
        }

    def _extract_structure_features(self, text: str) -> Dict[str, Any]:
        """Extract structural features (paragraphs, sentences)."""
        paragraphs = tokenize_paragraphs(text)
        paragraph_count = len(paragraphs)

        if paragraph_count == 0:
            return {
                "paragraph_count": 0,
                "avg_paragraph_length_sentences": 0.0,
                "avg_paragraph_length_words": 0.0,
                "single_sentence_paragraph_ratio": 0.0,
            }

        paragraph_sentence_counts = []
        paragraph_word_counts = []
        single_sentence_paragraphs = 0

        for para in paragraphs:
            sentences = tokenize_sentences(para)
            sentence_count = len(sentences)
            paragraph_sentence_counts.append(sentence_count)

            words = para.split()
            paragraph_word_counts.append(len(words))

            if sentence_count == 1:
                single_sentence_paragraphs += 1

        avg_sentences = sum(paragraph_sentence_counts) / paragraph_count
        avg_words = sum(paragraph_word_counts) / paragraph_count
        single_sentence_ratio = single_sentence_paragraphs / paragraph_count

        return {
            "paragraph_count": paragraph_count,
            "avg_paragraph_length_sentences": avg_sentences,
            "avg_paragraph_length_words": avg_words,
            "single_sentence_paragraph_ratio": single_sentence_ratio,
        }

    def _extract_capitalization_features(self, text: str) -> Dict[str, Any]:
        """Extract capitalization pattern features."""
        words = tokenize_words_preserve_case(text)

        if not words:
            return {
                "all_caps_word_count": 0,
                "all_caps_word_ratio": 0.0,
                "title_case_word_count": 0,
                "title_case_word_ratio": 0.0,
                "sentence_initial_cap_consistency": 0.0,
                "mixed_case_word_count": 0,
            }

        total_words = len(words)

        all_caps_words = [w for w in words if w.isupper() and len(w) > 1]
        all_caps_count = len(all_caps_words)

        title_case_words = [w for w in words if w.istitle() and len(w) > 1]
        title_case_count = len(title_case_words)

        mixed_case = []
        for word in words:
            if len(word) > 1:
                has_upper = any(c.isupper() for c in word)
                has_lower = any(c.islower() for c in word)
                not_title = not word.istitle()
                if has_upper and has_lower and not_title:
                    mixed_case.append(word)
        mixed_case_count = len(mixed_case)

        sentences = tokenize_sentences(text)
        properly_capitalized = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence[0].isupper():
                properly_capitalized += 1

        cap_consistency = safe_divide(properly_capitalized, len(sentences))

        return {
            "all_caps_word_count": all_caps_count,
            "all_caps_word_ratio": all_caps_count / total_words,
            "title_case_word_count": title_case_count,
            "title_case_word_ratio": title_case_count / total_words,
            "sentence_initial_cap_consistency": cap_consistency,
            "mixed_case_word_count": mixed_case_count,
        }

    def _extract_list_features(self, text: str) -> Dict[str, Any]:
        """Extract list and indentation features."""
        bullet_patterns = [
            r"^\s*[-*•]\s",
            r"^\s*[○◦▪▸►]\s",
        ]
        numbered_patterns = [
            r"^\s*\d+[.)]\s",
            r"^\s*[a-zA-Z][.)]\s",
            r"^\s*[ivxIVX]+[.)]\s",
        ]

        lines = text.split("\n")

        bullet_count = 0
        numbered_count = 0
        indentation_spaces = []

        for line in lines:
            for pattern in bullet_patterns:
                if re.match(pattern, line):
                    bullet_count += 1
                    break

            for pattern in numbered_patterns:
                if re.match(pattern, line):
                    numbered_count += 1
                    break

            leading_spaces = len(line) - len(line.lstrip(" "))
            if leading_spaces > 0:
                indentation_spaces.append(leading_spaces)

        avg_indentation = (
            sum(indentation_spaces) / len(indentation_spaces)
            if indentation_spaces
            else 0.0
        )

        return {
            "bullet_list_count": bullet_count,
            "numbered_list_count": numbered_count,
            "uses_bullet_lists": bullet_count > 0,
            "uses_numbered_lists": numbered_count > 0,
            "indentation_spaces_avg": avg_indentation,
        }

    def get_formatting_vector(self, text: str) -> List[float]:
        """
        Get a fixed-length vector of formatting features.

        Useful for ML models.
        """
        features = self.extract(text)

        vector_keys = [
            "multiple_space_ratio", "tab_ratio", "line_break_count",
            "double_line_break_count", "avg_line_length",
            "paragraph_count", "avg_paragraph_length_sentences",
            "avg_paragraph_length_words", "single_sentence_paragraph_ratio",
            "all_caps_word_ratio", "title_case_word_ratio",
            "sentence_initial_cap_consistency", "indentation_spaces_avg",
        ]

        vector = []
        for key in vector_keys:
            value = features.get(key, 0.0)
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            vector.append(float(value))

        return vector

    def analyze_formatting_style(self, text: str) -> Dict[str, Any]:
        """
        Provide human-readable analysis of formatting style.
        """
        features = self.extract(text)

        analysis = {
            "distinctive_patterns": [],
            "formality_level": "medium",
            "structure_style": "prose",
        }

        if features["all_caps_word_ratio"] > 0.05:
            analysis["distinctive_patterns"].append("Frequent ALL CAPS usage (emphatic)")
            analysis["formality_level"] = "informal"

        if features["uses_bullet_lists"] or features["uses_numbered_lists"]:
            analysis["distinctive_patterns"].append("Uses structured lists")
            analysis["structure_style"] = "structured"

        if features["single_sentence_paragraph_ratio"] > 0.5:
            analysis["distinctive_patterns"].append("Short paragraphs (punchy style)")

        if features["avg_paragraph_length_sentences"] > 5:
            analysis["distinctive_patterns"].append("Long paragraphs (academic/formal style)")
            analysis["formality_level"] = "formal"

        if features["sentence_initial_cap_consistency"] < 0.8:
            analysis["distinctive_patterns"].append("Inconsistent sentence capitalization")
            analysis["formality_level"] = "informal"

        if features["multiple_space_count"] > 5:
            analysis["distinctive_patterns"].append("Multiple space usage (possible copy-paste)")

        if features["tab_count"] > 0:
            analysis["distinctive_patterns"].append("Uses tabs for indentation")

        return analysis
