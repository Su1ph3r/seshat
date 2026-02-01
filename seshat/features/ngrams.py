"""
N-gram feature extraction for stylometric analysis.

Character and word n-grams are among the most robust features for
authorship attribution, capturing subconscious writing patterns.
"""

from collections import Counter
from typing import Dict, List, Any, Tuple
import re

from seshat.utils import tokenize_words, get_character_ngrams, get_word_ngrams


class NGramFeatures:
    """Extract n-gram features from text."""

    def __init__(
        self,
        char_ngram_range: Tuple[int, int] = (2, 5),
        word_ngram_range: Tuple[int, int] = (1, 3),
        max_features: int = 500,
    ):
        """
        Initialize n-gram extractor.

        Args:
            char_ngram_range: (min_n, max_n) for character n-grams
            word_ngram_range: (min_n, max_n) for word n-grams
            max_features: Maximum features to keep per n-gram type
        """
        self.char_ngram_range = char_ngram_range
        self.word_ngram_range = word_ngram_range
        self.max_features = max_features

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all n-gram features from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of n-gram features
        """
        if not text:
            return self._empty_features()

        features = {}

        char_features = self._extract_character_ngrams(text)
        features.update(char_features)

        words = tokenize_words(text)
        word_features = self._extract_word_ngrams(words)
        features.update(word_features)

        stats = self._extract_ngram_statistics(text, words)
        features.update(stats)

        return features

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary."""
        return {
            "char_2gram_entropy": 0.0,
            "char_3gram_entropy": 0.0,
            "char_4gram_entropy": 0.0,
            "char_5gram_entropy": 0.0,
            "word_1gram_entropy": 0.0,
            "word_2gram_entropy": 0.0,
            "word_3gram_entropy": 0.0,
            "char_2gram_unique_ratio": 0.0,
            "char_3gram_unique_ratio": 0.0,
            "char_4gram_unique_ratio": 0.0,
            "word_2gram_unique_ratio": 0.0,
            "top_char_2grams": {},
            "top_char_3grams": {},
            "top_char_4grams": {},
            "top_word_2grams": {},
        }

    def _extract_character_ngrams(self, text: str) -> Dict[str, Any]:
        """Extract character n-gram features."""
        features = {}

        text_lower = text.lower()

        for n in range(self.char_ngram_range[0], self.char_ngram_range[1] + 1):
            ngrams = get_character_ngrams(text_lower, n)

            if not ngrams:
                features[f"char_{n}gram_entropy"] = 0.0
                features[f"char_{n}gram_unique_ratio"] = 0.0
                features[f"top_char_{n}grams"] = {}
                continue

            ngram_counts = Counter(ngrams)
            total = len(ngrams)
            unique = len(ngram_counts)

            entropy = self._calculate_entropy(ngram_counts, total)
            features[f"char_{n}gram_entropy"] = entropy

            features[f"char_{n}gram_unique_ratio"] = unique / total

            top_ngrams = ngram_counts.most_common(min(50, self.max_features))
            features[f"top_char_{n}grams"] = {
                ng: count / total for ng, count in top_ngrams
            }

        return features

    def _extract_word_ngrams(self, words: List[str]) -> Dict[str, Any]:
        """Extract word n-gram features."""
        features = {}

        if not words:
            for n in range(self.word_ngram_range[0], self.word_ngram_range[1] + 1):
                features[f"word_{n}gram_entropy"] = 0.0
                features[f"word_{n}gram_unique_ratio"] = 0.0
                if n > 1:
                    features[f"top_word_{n}grams"] = {}
            return features

        for n in range(self.word_ngram_range[0], self.word_ngram_range[1] + 1):
            if n == 1:
                ngrams = [(w,) for w in words]
            else:
                ngrams = get_word_ngrams(words, n)

            if not ngrams:
                features[f"word_{n}gram_entropy"] = 0.0
                features[f"word_{n}gram_unique_ratio"] = 0.0
                if n > 1:
                    features[f"top_word_{n}grams"] = {}
                continue

            ngram_counts = Counter(ngrams)
            total = len(ngrams)
            unique = len(ngram_counts)

            entropy = self._calculate_entropy(ngram_counts, total)
            features[f"word_{n}gram_entropy"] = entropy

            features[f"word_{n}gram_unique_ratio"] = unique / total

            if n > 1:
                top_ngrams = ngram_counts.most_common(min(50, self.max_features))
                features[f"top_word_{n}grams"] = {
                    " ".join(ng): count / total for ng, count in top_ngrams
                }

        return features

    def _extract_ngram_statistics(
        self, text: str, words: List[str]
    ) -> Dict[str, Any]:
        """Extract statistical features about n-gram distributions."""
        features = {}

        text_lower = text.lower()
        char_4grams = get_character_ngrams(text_lower, 4)

        if char_4grams:
            counts = Counter(char_4grams)

            hapax = sum(1 for c in counts.values() if c == 1)
            features["char_4gram_hapax_ratio"] = hapax / len(counts)

            values = list(counts.values())
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            features["char_4gram_frequency_variance"] = variance
        else:
            features["char_4gram_hapax_ratio"] = 0.0
            features["char_4gram_frequency_variance"] = 0.0

        if len(words) >= 2:
            word_bigrams = get_word_ngrams(words, 2)
            counts = Counter(word_bigrams)

            hapax = sum(1 for c in counts.values() if c == 1)
            features["word_2gram_hapax_ratio"] = hapax / len(counts) if counts else 0.0
        else:
            features["word_2gram_hapax_ratio"] = 0.0

        return features

    def _calculate_entropy(self, counts: Counter, total: int) -> float:
        """Calculate Shannon entropy of n-gram distribution."""
        import math

        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)

        return entropy

    def get_character_ngram_vector(
        self, text: str, n: int, vocabulary: List[str]
    ) -> List[float]:
        """
        Get a fixed-length character n-gram frequency vector.

        Args:
            text: Input text
            n: N-gram size
            vocabulary: Ordered list of n-grams to include

        Returns:
            Frequency vector aligned with vocabulary
        """
        text_lower = text.lower()
        ngrams = get_character_ngrams(text_lower, n)

        if not ngrams:
            return [0.0] * len(vocabulary)

        counts = Counter(ngrams)
        total = len(ngrams)

        return [counts.get(ng, 0) / total for ng in vocabulary]

    def get_word_ngram_vector(
        self, words: List[str], n: int, vocabulary: List[Tuple[str, ...]]
    ) -> List[float]:
        """
        Get a fixed-length word n-gram frequency vector.

        Args:
            words: List of words
            n: N-gram size
            vocabulary: Ordered list of n-gram tuples

        Returns:
            Frequency vector aligned with vocabulary
        """
        if len(words) < n:
            return [0.0] * len(vocabulary)

        ngrams = get_word_ngrams(words, n)
        counts = Counter(ngrams)
        total = len(ngrams)

        return [counts.get(ng, 0) / total for ng in vocabulary]

    def build_vocabulary(
        self, texts: List[str], n: int, ngram_type: str = "char", min_freq: int = 2
    ) -> List[Any]:
        """
        Build a vocabulary of n-grams from a corpus.

        Args:
            texts: List of texts
            n: N-gram size
            ngram_type: "char" or "word"
            min_freq: Minimum frequency to include

        Returns:
            List of n-grams meeting the frequency threshold
        """
        all_counts: Counter = Counter()

        for text in texts:
            if ngram_type == "char":
                ngrams = get_character_ngrams(text.lower(), n)
            else:
                words = tokenize_words(text)
                ngrams = get_word_ngrams(words, n)

            all_counts.update(ngrams)

        vocabulary = [
            ng for ng, count in all_counts.most_common()
            if count >= min_freq
        ]

        return vocabulary[:self.max_features]

    def get_most_distinctive_ngrams(
        self,
        text: str,
        reference_texts: List[str],
        n: int = 4,
        ngram_type: str = "char",
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find n-grams that distinguish this text from reference texts.

        Args:
            text: Target text
            reference_texts: Corpus of reference texts
            n: N-gram size
            ngram_type: "char" or "word"
            top_k: Number of distinctive n-grams to return

        Returns:
            List of distinctive n-grams with scores
        """
        if ngram_type == "char":
            target_ngrams = Counter(get_character_ngrams(text.lower(), n))
        else:
            words = tokenize_words(text)
            target_ngrams = Counter(get_word_ngrams(words, n))

        ref_ngrams: Counter = Counter()
        for ref_text in reference_texts:
            if ngram_type == "char":
                ref_ngrams.update(get_character_ngrams(ref_text.lower(), n))
            else:
                words = tokenize_words(ref_text)
                ref_ngrams.update(get_word_ngrams(words, n))

        target_total = sum(target_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        if target_total == 0 or ref_total == 0:
            return []

        distinctive = []
        for ngram, count in target_ngrams.items():
            target_freq = count / target_total
            ref_freq = ref_ngrams.get(ngram, 0) / ref_total

            if ref_freq == 0:
                score = target_freq * 10
            else:
                score = target_freq / ref_freq

            distinctive.append({
                "ngram": ngram if ngram_type == "char" else " ".join(ngram),
                "target_freq": target_freq,
                "reference_freq": ref_freq,
                "distinctiveness_score": score,
            })

        distinctive.sort(key=lambda x: x["distinctiveness_score"], reverse=True)
        return distinctive[:top_k]


# Alias for backward compatibility
NGramExtractor = NGramFeatures
