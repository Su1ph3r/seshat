"""
Adversarial detection for identifying deliberate style obfuscation.

Detects when an author is intentionally trying to disguise their writing style
or imitate another author's style.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from collections import Counter

from seshat.utils import tokenize_words, tokenize_sentences
from seshat.analyzer import Analyzer


class AdversarialDetector:
    """
    Detect adversarial stylometric manipulation.

    Identifies:
    - Deliberate obfuscation attempts
    - Style imitation
    - Machine translation artifacts
    """

    def __init__(self, analyzer: Optional[Analyzer] = None):
        """
        Initialize adversarial detector.

        Args:
            analyzer: Analyzer instance
        """
        self.analyzer = analyzer or Analyzer()

    def detect_obfuscation(self, text: str) -> Dict[str, Any]:
        """
        Detect if text shows signs of deliberate style obfuscation.

        Args:
            text: Input text to analyze

        Returns:
            Obfuscation detection results
        """
        if not text or len(text) < 200:
            return {
                "is_obfuscated": False,
                "confidence": "low",
                "reason": "Insufficient text for analysis",
            }

        indicators = []
        obfuscation_score = 0.0

        consistency_result = self._check_consistency(text)
        if consistency_result["is_suspicious"]:
            indicators.extend(consistency_result["indicators"])
            obfuscation_score += consistency_result["score"]

        vocabulary_result = self._check_vocabulary_anomalies(text)
        if vocabulary_result["is_suspicious"]:
            indicators.extend(vocabulary_result["indicators"])
            obfuscation_score += vocabulary_result["score"]

        pattern_result = self._check_pattern_disruption(text)
        if pattern_result["is_suspicious"]:
            indicators.extend(pattern_result["indicators"])
            obfuscation_score += pattern_result["score"]

        is_obfuscated = obfuscation_score > 0.5

        if obfuscation_score > 0.8:
            confidence = "high"
        elif obfuscation_score > 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "is_obfuscated": is_obfuscated,
            "confidence": confidence,
            "obfuscation_score": obfuscation_score,
            "indicators": indicators,
            "detailed_analysis": {
                "consistency": consistency_result,
                "vocabulary": vocabulary_result,
                "patterns": pattern_result,
            },
        }

    def _check_consistency(self, text: str) -> Dict[str, Any]:
        """Check for unnatural consistency (sign of deliberate control)."""
        words = tokenize_words(text)
        sentences = tokenize_sentences(text)

        indicators = []
        score = 0.0

        word_lengths = [len(w) for w in words]
        if word_lengths:
            cv = np.std(word_lengths) / np.mean(word_lengths)
            if cv < 0.3:
                indicators.append("Unnaturally consistent word lengths")
                score += 0.2

        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            cv = np.std(sentence_lengths) / (np.mean(sentence_lengths) + 0.001)
            if cv < 0.2:
                indicators.append("Unnaturally consistent sentence lengths")
                score += 0.2

        word_counts = Counter(words)
        if word_counts:
            frequencies = list(word_counts.values())
            freq_cv = np.std(frequencies) / (np.mean(frequencies) + 0.001)
            if freq_cv < 0.5:
                indicators.append("Abnormally flat word frequency distribution")
                score += 0.15

        return {
            "is_suspicious": len(indicators) > 0,
            "indicators": indicators,
            "score": score,
        }

    def _check_vocabulary_anomalies(self, text: str) -> Dict[str, Any]:
        """Check for vocabulary that seems forced or unnatural."""
        words = tokenize_words(text)

        indicators = []
        score = 0.0

        word_counts = Counter(words)
        unique_words = len(word_counts)
        total_words = len(words)

        ttr = unique_words / total_words if total_words > 0 else 0

        if ttr > 0.8:
            indicators.append("Unusually high vocabulary diversity (possible thesaurus use)")
            score += 0.2

        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        if avg_word_length > 7:
            indicators.append("Unusually long average word length")
            score += 0.15

        rare_words = [w for w in words if len(w) > 10]
        rare_ratio = len(rare_words) / len(words) if words else 0
        if rare_ratio > 0.1:
            indicators.append("High proportion of rare/long words")
            score += 0.15

        return {
            "is_suspicious": len(indicators) > 0,
            "indicators": indicators,
            "score": score,
            "metrics": {
                "ttr": ttr,
                "avg_word_length": avg_word_length,
                "rare_word_ratio": rare_ratio,
            },
        }

    def _check_pattern_disruption(self, text: str) -> Dict[str, Any]:
        """Check for signs of deliberately disrupted patterns."""
        sentences = tokenize_sentences(text)

        indicators = []
        score = 0.0

        starters = []
        for sentence in sentences:
            words = sentence.strip().split()
            if words:
                starters.append(words[0].lower())

        if starters:
            starter_counts = Counter(starters)
            if len(starter_counts) == len(starters):
                indicators.append("Every sentence starts with a different word (possible obfuscation)")
                score += 0.15

        punctuation_counts = []
        for sentence in sentences:
            p_count = sum(1 for c in sentence if c in ".,;:!?")
            punctuation_counts.append(p_count)

        if punctuation_counts:
            p_cv = np.std(punctuation_counts) / (np.mean(punctuation_counts) + 0.001)
            if p_cv < 0.2 and np.mean(punctuation_counts) > 2:
                indicators.append("Unnaturally consistent punctuation patterns")
                score += 0.15

        return {
            "is_suspicious": len(indicators) > 0,
            "indicators": indicators,
            "score": score,
        }

    def detect_imitation(
        self,
        text: str,
        target_profile: Any,
    ) -> Dict[str, Any]:
        """
        Detect if text is attempting to imitate a specific author.

        Args:
            text: Text to analyze
            target_profile: AuthorProfile of the author being imitated

        Returns:
            Imitation detection results
        """
        analysis = self.analyzer.analyze(text)
        text_features = analysis.get_flat_features()

        profile_features = target_profile.aggregated_features
        profile_stats = target_profile.feature_statistics

        very_close = []
        suspicious_matches = []

        for key, value in text_features.items():
            if key in profile_stats:
                stats = profile_stats[key]
                mean = stats["mean"]
                std = stats["std"]

                if std > 0:
                    z_score = abs(value - mean) / std

                    if z_score < 0.1:
                        very_close.append(key)

        very_close_ratio = len(very_close) / len(text_features) if text_features else 0

        is_imitation = False
        if very_close_ratio > 0.5:
            obfuscation_check = self.detect_obfuscation(text)
            if obfuscation_check["is_obfuscated"]:
                is_imitation = True
                suspicious_matches.append("High match ratio combined with obfuscation signals")

        if is_imitation:
            confidence = "medium" if very_close_ratio > 0.7 else "low"
        else:
            confidence = "low"

        return {
            "is_imitation": is_imitation,
            "confidence": confidence,
            "match_ratio": very_close_ratio,
            "very_close_features": len(very_close),
            "suspicious_patterns": suspicious_matches,
        }

    def detect_translation(self, text: str) -> Dict[str, Any]:
        """
        Detect if text shows signs of machine translation.

        Args:
            text: Text to analyze

        Returns:
            Translation detection results
        """
        words = tokenize_words(text)
        sentences = tokenize_sentences(text)

        indicators = []
        score = 0.0

        idiom_markers = [
            "piece of cake", "break a leg", "hit the hay", "under the weather",
            "raining cats and dogs", "kick the bucket", "spill the beans",
        ]

        text_lower = text.lower()
        idioms_found = sum(1 for idiom in idiom_markers if idiom in text_lower)
        if idioms_found == 0 and len(words) > 200:
            indicators.append("No idiomatic expressions (possible translation)")
            score += 0.15

        contraction_count = sum(1 for w in words if "'" in w and w not in ["'s", "'t", "'d", "'m", "'re", "'ve", "'ll"])
        actual_contractions = sum(1 for w in words if w in ["don't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't", "didn't", "couldn't", "wouldn't", "shouldn't"])

        if actual_contractions == 0 and len(words) > 100:
            indicators.append("No contractions (formal, possibly translated)")
            score += 0.1

        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            avg_length = np.mean(sentence_lengths)
            if avg_length > 25:
                indicators.append("Very long average sentence length (translation artifact)")
                score += 0.1

        is_translated = score > 0.25

        return {
            "is_translated": is_translated,
            "confidence": "medium" if score > 0.35 else "low",
            "translation_score": score,
            "indicators": indicators,
        }
