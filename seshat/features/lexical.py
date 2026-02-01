"""
Lexical feature extraction for stylometric analysis.

Includes vocabulary richness metrics, word-level statistics, and word choice patterns.
"""

import math
from collections import Counter
from typing import Dict, List, Any

from seshat.utils import (
    tokenize_words,
    tokenize_words_preserve_case,
    get_word_length_stats,
    is_contraction,
    is_abbreviation,
    safe_divide,
)


class LexicalFeatures:
    """Extract lexical features from text."""

    def __init__(self):
        self.foreign_words = self._load_foreign_words()
        self.slang_words = self._load_slang_words()
        self.formal_words = self._load_formal_words()

    def _load_foreign_words(self) -> set:
        """Load common foreign words used in English text."""
        return {
            "je", "ne", "sais", "quoi", "cest", "la", "vie", "raison", "detre",
            "zeitgeist", "schadenfreude", "wanderlust", "kindergarten", "gesundheit",
            "carpe", "diem", "per", "se", "vice", "versa", "et", "cetera", "ad",
            "hoc", "bona", "fide", "quid", "pro", "quo", "modus", "operandi",
            "status", "quo", "de", "facto", "in", "situ", "ex", "nihilo",
            "hasta", "manana", "siesta", "fiesta", "gracias", "amigo",
            "ciao", "bella", "dolce", "vita", "cappuccino", "espresso",
            "karaoke", "tsunami", "origami", "samurai", "anime", "manga",
        }

    def _load_slang_words(self) -> set:
        """Load common slang words."""
        return {
            "gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "lemme",
            "gimme", "whatcha", "gotcha", "aint", "yall", "yep", "nope",
            "yeah", "yea", "nah", "meh", "dude", "bro", "bruh", "fam",
            "lit", "fire", "sick", "dope", "epic", "legit", "lowkey", "highkey",
            "salty", "savage", "shook", "woke", "slay", "stan", "vibe", "mood",
            "flex", "ghosting", "sus", "cap", "nocap", "bussin", "bet", "finna",
            "deadass", "periodt", "tea", "sis", "queen", "king", "goat",
        }

    def _load_formal_words(self) -> set:
        """Load formal/academic vocabulary."""
        return {
            "furthermore", "moreover", "nevertheless", "nonetheless", "however",
            "therefore", "consequently", "subsequently", "accordingly", "hence",
            "whereby", "wherein", "whereas", "notwithstanding", "hitherto",
            "heretofore", "aforementioned", "herein", "thereof", "therein",
            "pursuant", "regarding", "concerning", "pertaining", "respective",
            "preliminary", "subsequent", "prior", "aforementioned", "foregoing",
            "comprising", "constituting", "encompassing", "facilitating",
            "implementing", "indicating", "demonstrating", "illustrating",
            "exemplifying", "substantiating", "corroborating", "elucidating",
        }

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all lexical features from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of lexical features
        """
        words = tokenize_words(text)
        words_preserve_case = tokenize_words_preserve_case(text)

        if not words:
            return self._empty_features()

        features = {}

        vocabulary_features = self._extract_vocabulary_richness(words)
        features.update(vocabulary_features)

        word_level_features = self._extract_word_level_stats(words)
        features.update(word_level_features)

        word_choice_features = self._extract_word_choice_patterns(words, words_preserve_case)
        features.update(word_choice_features)

        return features

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary for empty text."""
        return {
            "total_words": 0,
            "unique_words": 0,
            "type_token_ratio": 0.0,
            "hapax_legomena_count": 0,
            "hapax_legomena_ratio": 0.0,
            "hapax_dislegomena_count": 0,
            "hapax_dislegomena_ratio": 0.0,
            "yules_k": 0.0,
            "simpsons_d": 0.0,
            "honores_r": 0.0,
            "avg_word_length": 0.0,
            "word_length_std": 0.0,
            "word_length_min": 0,
            "word_length_max": 0,
            "short_word_ratio": 0.0,
            "long_word_ratio": 0.0,
            "contraction_ratio": 0.0,
            "abbreviation_ratio": 0.0,
            "slang_ratio": 0.0,
            "formal_ratio": 0.0,
            "foreign_word_ratio": 0.0,
            "word_length_distribution": {},
        }

    def _extract_vocabulary_richness(self, words: List[str]) -> Dict[str, Any]:
        """
        Extract vocabulary richness metrics.

        Includes:
        - Type-Token Ratio (TTR)
        - Hapax Legomena (words appearing once)
        - Hapax Dislegomena (words appearing twice)
        - Yule's K characteristic
        - Simpson's D diversity index
        - Honore's R statistic
        """
        total_words = len(words)
        word_counts = Counter(words)
        unique_words = len(word_counts)

        ttr = safe_divide(unique_words, total_words)

        hapax_legomena = [w for w, c in word_counts.items() if c == 1]
        hapax_legomena_count = len(hapax_legomena)
        hapax_legomena_ratio = safe_divide(hapax_legomena_count, total_words)

        hapax_dislegomena = [w for w, c in word_counts.items() if c == 2]
        hapax_dislegomena_count = len(hapax_dislegomena)
        hapax_dislegomena_ratio = safe_divide(hapax_dislegomena_count, total_words)

        yules_k = self._calculate_yules_k(word_counts, total_words)

        simpsons_d = self._calculate_simpsons_d(word_counts, total_words)

        honores_r = self._calculate_honores_r(
            total_words, unique_words, hapax_legomena_count
        )

        return {
            "total_words": total_words,
            "unique_words": unique_words,
            "type_token_ratio": ttr,
            "hapax_legomena_count": hapax_legomena_count,
            "hapax_legomena_ratio": hapax_legomena_ratio,
            "hapax_dislegomena_count": hapax_dislegomena_count,
            "hapax_dislegomena_ratio": hapax_dislegomena_ratio,
            "yules_k": yules_k,
            "simpsons_d": simpsons_d,
            "honores_r": honores_r,
        }

    def _calculate_yules_k(self, word_counts: Counter, total_words: int) -> float:
        """
        Calculate Yule's K characteristic.

        Higher K indicates less diversity (more repetition).
        Formula: K = 10^4 * (M2 - N) / N^2
        where M2 = sum(r^2 * V(r)) and V(r) = number of words appearing r times
        """
        if total_words <= 1:
            return 0.0

        frequency_spectrum = Counter(word_counts.values())

        m2 = sum((freq ** 2) * count for freq, count in frequency_spectrum.items())

        k = 10000 * (m2 - total_words) / (total_words ** 2)

        return max(k, 0.0)

    def _calculate_simpsons_d(self, word_counts: Counter, total_words: int) -> float:
        """
        Calculate Simpson's D diversity index.

        Higher D indicates less diversity.
        Formula: D = sum(n_i * (n_i - 1)) / (N * (N - 1))
        """
        if total_words <= 1:
            return 0.0

        numerator = sum(count * (count - 1) for count in word_counts.values())
        denominator = total_words * (total_words - 1)

        return safe_divide(numerator, denominator)

    def _calculate_honores_r(
        self, total_words: int, unique_words: int, hapax_count: int
    ) -> float:
        """
        Calculate Honore's R statistic.

        Higher R indicates richer vocabulary.
        Formula: R = 100 * log(N) / (1 - V1/V)
        where N = total words, V = unique words, V1 = hapax legomena
        """
        if total_words <= 1 or unique_words == 0:
            return 0.0

        ratio = safe_divide(hapax_count, unique_words)

        if ratio >= 1.0:
            return 0.0

        denominator = 1 - ratio
        if denominator <= 0:
            return 0.0

        r = 100 * math.log(total_words) / denominator

        return r

    def _extract_word_level_stats(self, words: List[str]) -> Dict[str, Any]:
        """Extract word length statistics."""
        stats = get_word_length_stats(words)

        length_distribution = Counter(len(w) for w in words)
        total = len(words)
        length_distribution_normalized = {
            str(length): count / total
            for length, count in sorted(length_distribution.items())
        }

        return {
            "avg_word_length": stats["mean"],
            "word_length_std": stats["std"],
            "word_length_min": stats["min"],
            "word_length_max": stats["max"],
            "short_word_ratio": stats["short_ratio"],
            "long_word_ratio": stats["long_ratio"],
            "word_length_distribution": length_distribution_normalized,
        }

    def _extract_word_choice_patterns(
        self, words: List[str], words_preserve_case: List[str]
    ) -> Dict[str, Any]:
        """
        Extract word choice pattern features.

        Includes contractions, abbreviations, slang, formal vocabulary, foreign words.
        """
        total_words = len(words)
        if total_words == 0:
            return {
                "contraction_ratio": 0.0,
                "abbreviation_ratio": 0.0,
                "slang_ratio": 0.0,
                "formal_ratio": 0.0,
                "foreign_word_ratio": 0.0,
            }

        contractions = sum(1 for w in words if is_contraction(w))
        abbreviations = sum(1 for w in words if is_abbreviation(w))
        slang_count = sum(1 for w in words if w in self.slang_words)
        formal_count = sum(1 for w in words if w in self.formal_words)
        foreign_count = sum(1 for w in words if w in self.foreign_words)

        return {
            "contraction_ratio": contractions / total_words,
            "abbreviation_ratio": abbreviations / total_words,
            "slang_ratio": slang_count / total_words,
            "formal_ratio": formal_count / total_words,
            "foreign_word_ratio": foreign_count / total_words,
        }

    def get_top_words(self, text: str, n: int = 20) -> List[tuple]:
        """Get the n most frequent words in text."""
        words = tokenize_words(text)
        return Counter(words).most_common(n)

    def get_rare_words(self, text: str, threshold: int = 1) -> List[str]:
        """Get words appearing at most 'threshold' times."""
        words = tokenize_words(text)
        word_counts = Counter(words)
        return [word for word, count in word_counts.items() if count <= threshold]


# Alias for backward compatibility
LexicalFeatureExtractor = LexicalFeatures
