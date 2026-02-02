"""
Advanced metrics for personality disorder analysis.

Provides temporal pattern analysis, linguistic complexity metrics,
and response style indicators.
"""

import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter

from seshat.utils import tokenize_words

from .pd_dictionaries import (
    TEMPORAL_MARKERS,
    HEDGING_WORDS,
    ABSOLUTIST_WORDS,
    DEFLECTION_PHRASES,
    COMPLEX_VOCABULARY,
    SIMPLE_VOCABULARY,
)


@dataclass
class TemporalProfile:
    """Temporal focus analysis results."""
    past_focus: float  # 0.0 to 1.0
    present_focus: float  # 0.0 to 1.0
    future_focus: float  # 0.0 to 1.0
    dominant_focus: str  # "past", "present", "future", or "balanced"
    temporal_markers_found: Dict[str, int]
    interpretation: str


@dataclass
class ComplexityMetrics:
    """Linguistic complexity analysis results."""
    vocabulary_sophistication: float  # 0.0 to 1.0
    lexical_diversity: float  # Type-token ratio
    avg_word_length: float
    avg_sentence_length: float
    complex_word_ratio: float  # Words with 3+ syllables
    readability_score: float  # Approximation of grade level
    interpretation: str


@dataclass
class ResponseStyleMetrics:
    """Response style analysis results."""
    hedging_ratio: float  # Tentative language
    absolutism_ratio: float  # Black-and-white language
    deflection_ratio: float  # Redirecting/avoiding language
    self_reference_ratio: float  # First-person pronouns
    other_reference_ratio: float  # References to others
    emotional_expressiveness: float  # Emotion word ratio
    certainty_ratio: float  # Certain vs uncertain language
    interpretation: str


class PDAdvancedMetrics:
    """Advanced linguistic metrics for personality disorder analysis."""

    def __init__(self):
        """Initialize advanced metrics analyzer."""
        self.temporal_markers = TEMPORAL_MARKERS
        self.hedging_words = HEDGING_WORDS
        self.absolutist_words = ABSOLUTIST_WORDS
        self.deflection_phrases = DEFLECTION_PHRASES
        self.complex_vocab = set(COMPLEX_VOCABULARY)
        self.simple_vocab = set(SIMPLE_VOCABULARY)

        # Emotion words for expressiveness
        self.emotion_words = {
            "happy", "sad", "angry", "afraid", "disgusted", "surprised",
            "joy", "fear", "love", "hate", "anxious", "excited", "depressed",
            "frustrated", "irritated", "terrified", "delighted", "miserable",
            "furious", "thrilled", "hopeful", "hopeless", "guilty", "ashamed",
            "proud", "jealous", "envious", "lonely", "grateful", "content",
            "hurt", "heartbroken", "devastated", "elated", "worried", "nervous",
        }

    def analyze_temporal_patterns(self, text: str) -> TemporalProfile:
        """
        Analyze temporal focus in the text.

        Args:
            text: Input text to analyze

        Returns:
            TemporalProfile with temporal orientation analysis
        """
        if not text:
            return self._empty_temporal_profile()

        text_lower = text.lower()
        words = tokenize_words(text)
        word_count = len(words)

        if word_count == 0:
            return self._empty_temporal_profile()

        markers_found = {"past": 0, "present": 0, "future": 0}

        # Count temporal markers
        for temporal_type, markers in self.temporal_markers.items():
            for marker in markers:
                if ' ' in marker:
                    # Multi-word marker
                    markers_found[temporal_type] += text_lower.count(marker.lower())
                else:
                    # Single word
                    markers_found[temporal_type] += sum(1 for w in words if w == marker.lower())

        total_markers = sum(markers_found.values())

        if total_markers == 0:
            return TemporalProfile(
                past_focus=0.33,
                present_focus=0.34,
                future_focus=0.33,
                dominant_focus="balanced",
                temporal_markers_found=markers_found,
                interpretation="No clear temporal orientation detected",
            )

        # Calculate focus ratios
        past_focus = markers_found["past"] / total_markers
        present_focus = markers_found["present"] / total_markers
        future_focus = markers_found["future"] / total_markers

        # Determine dominant focus
        max_focus = max(past_focus, present_focus, future_focus)
        if max_focus < 0.4:  # No clear dominant focus
            dominant_focus = "balanced"
        elif past_focus == max_focus:
            dominant_focus = "past"
        elif present_focus == max_focus:
            dominant_focus = "present"
        else:
            dominant_focus = "future"

        # Generate interpretation
        interpretations = {
            "past": "Past-focused language may indicate rumination or reflection",
            "present": "Present-focused language indicates here-and-now orientation",
            "future": "Future-focused language suggests planning or anticipation",
            "balanced": "Balanced temporal orientation across past, present, and future",
        }

        return TemporalProfile(
            past_focus=past_focus,
            present_focus=present_focus,
            future_focus=future_focus,
            dominant_focus=dominant_focus,
            temporal_markers_found=markers_found,
            interpretation=interpretations[dominant_focus],
        )

    def analyze_linguistic_complexity(self, text: str) -> ComplexityMetrics:
        """
        Analyze linguistic complexity of the text.

        Args:
            text: Input text to analyze

        Returns:
            ComplexityMetrics with vocabulary and structural analysis
        """
        if not text:
            return self._empty_complexity_metrics()

        words = tokenize_words(text)
        word_count = len(words)

        if word_count == 0:
            return self._empty_complexity_metrics()

        # Calculate lexical diversity (type-token ratio)
        unique_words = set(words)
        lexical_diversity = len(unique_words) / word_count

        # Average word length
        avg_word_length = sum(len(w) for w in words) / word_count

        # Count complex words (3+ syllables)
        complex_word_count = sum(1 for w in words if self._count_syllables(w) >= 3)
        complex_word_ratio = complex_word_count / word_count

        # Vocabulary sophistication
        sophisticated_count = sum(1 for w in words if w in self.complex_vocab)
        simple_count = sum(1 for w in words if w in self.simple_vocab)
        vocab_balance = sophisticated_count - simple_count
        vocabulary_sophistication = min(1.0, max(0.0, 0.5 + (vocab_balance / word_count) * 10))

        # Sentence analysis
        sentences = self._split_sentences(text)
        sentence_count = len(sentences) if sentences else 1
        avg_sentence_length = word_count / sentence_count

        # Readability approximation (simplified Flesch-Kincaid)
        # 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        total_syllables = sum(self._count_syllables(w) for w in words)
        syllables_per_word = total_syllables / word_count if word_count > 0 else 0
        readability_score = max(0, 0.39 * avg_sentence_length + 11.8 * syllables_per_word - 15.59)

        # Generate interpretation
        if vocabulary_sophistication > 0.7:
            interpretation = "High linguistic sophistication with complex vocabulary"
        elif vocabulary_sophistication < 0.3:
            interpretation = "Simple, accessible language"
        else:
            interpretation = "Moderate linguistic complexity"

        return ComplexityMetrics(
            vocabulary_sophistication=vocabulary_sophistication,
            lexical_diversity=lexical_diversity,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            complex_word_ratio=complex_word_ratio,
            readability_score=readability_score,
            interpretation=interpretation,
        )

    def analyze_response_style(self, text: str) -> ResponseStyleMetrics:
        """
        Analyze response style indicators.

        Args:
            text: Input text to analyze

        Returns:
            ResponseStyleMetrics with hedging, absolutism, and other style indicators
        """
        if not text:
            return self._empty_response_style()

        text_lower = text.lower()
        words = tokenize_words(text)
        word_count = len(words)

        if word_count == 0:
            return self._empty_response_style()

        # Hedging ratio
        hedging_count = 0
        for hedge in self.hedging_words:
            if ' ' in hedge:
                hedging_count += text_lower.count(hedge.lower())
            else:
                hedging_count += sum(1 for w in words if w == hedge.lower())
        hedging_ratio = min(1.0, hedging_count / word_count * 10)

        # Absolutism ratio
        absolutism_count = 0
        for absolute in self.absolutist_words:
            absolutism_count += sum(1 for w in words if w == absolute.lower())
        absolutism_ratio = min(1.0, absolutism_count / word_count * 10)

        # Deflection ratio
        deflection_count = 0
        for phrase in self.deflection_phrases:
            deflection_count += text_lower.count(phrase.lower())
        deflection_ratio = min(1.0, deflection_count / word_count * 20)

        # Self-reference ratio (first-person pronouns)
        first_person = {'i', "i'm", "i've", "i'd", "i'll", 'me', 'my', 'mine', 'myself'}
        self_ref_count = sum(1 for w in words if w in first_person)
        self_reference_ratio = self_ref_count / word_count

        # Other-reference ratio (third-person)
        third_person = {'he', 'she', 'they', 'him', 'her', 'them', 'his', 'their'}
        other_ref_count = sum(1 for w in words if w in third_person)
        other_reference_ratio = other_ref_count / word_count

        # Emotional expressiveness
        emotion_count = sum(1 for w in words if w in self.emotion_words)
        emotional_expressiveness = min(1.0, emotion_count / word_count * 20)

        # Certainty ratio (absolutism vs hedging)
        certainty_ratio = (absolutism_ratio - hedging_ratio + 1) / 2  # Normalize to 0-1

        # Generate interpretation
        style_characteristics = []
        if hedging_ratio > 0.3:
            style_characteristics.append("tentative")
        if absolutism_ratio > 0.3:
            style_characteristics.append("absolute/black-and-white")
        if deflection_ratio > 0.2:
            style_characteristics.append("deflecting")
        if self_reference_ratio > 0.1:
            style_characteristics.append("self-focused")
        if emotional_expressiveness > 0.3:
            style_characteristics.append("emotionally expressive")

        if style_characteristics:
            interpretation = f"Response style: {', '.join(style_characteristics)}"
        else:
            interpretation = "Neutral response style"

        return ResponseStyleMetrics(
            hedging_ratio=hedging_ratio,
            absolutism_ratio=absolutism_ratio,
            deflection_ratio=deflection_ratio,
            self_reference_ratio=self_reference_ratio,
            other_reference_ratio=other_reference_ratio,
            emotional_expressiveness=emotional_expressiveness,
            certainty_ratio=certainty_ratio,
            interpretation=interpretation,
        )

    def _count_syllables(self, word: str) -> int:
        """
        Estimate syllable count for a word.

        Uses a simple vowel-counting heuristic.
        """
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        prev_is_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel

        # Adjust for silent e
        if word.endswith('e'):
            count -= 1

        # Ensure at least 1 syllable
        return max(1, count)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _empty_temporal_profile(self) -> TemporalProfile:
        """Return empty temporal profile."""
        return TemporalProfile(
            past_focus=0.0,
            present_focus=0.0,
            future_focus=0.0,
            dominant_focus="unknown",
            temporal_markers_found={"past": 0, "present": 0, "future": 0},
            interpretation="Insufficient text for temporal analysis",
        )

    def _empty_complexity_metrics(self) -> ComplexityMetrics:
        """Return empty complexity metrics."""
        return ComplexityMetrics(
            vocabulary_sophistication=0.0,
            lexical_diversity=0.0,
            avg_word_length=0.0,
            avg_sentence_length=0.0,
            complex_word_ratio=0.0,
            readability_score=0.0,
            interpretation="Insufficient text for complexity analysis",
        )

    def _empty_response_style(self) -> ResponseStyleMetrics:
        """Return empty response style metrics."""
        return ResponseStyleMetrics(
            hedging_ratio=0.0,
            absolutism_ratio=0.0,
            deflection_ratio=0.0,
            self_reference_ratio=0.0,
            other_reference_ratio=0.0,
            emotional_expressiveness=0.0,
            certainty_ratio=0.5,
            interpretation="Insufficient text for response style analysis",
        )

    def get_pd_relevant_metrics(
        self,
        temporal: TemporalProfile,
        complexity: ComplexityMetrics,
        response_style: ResponseStyleMetrics,
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract metrics particularly relevant to personality disorder assessment.

        Args:
            temporal: Temporal profile analysis
            complexity: Complexity metrics
            response_style: Response style metrics

        Returns:
            Dictionary with disorder-relevant metric summaries
        """
        metrics = {}

        # Paranoid-relevant: past focus (rumination), other-reference (blame)
        metrics["paranoid_relevant"] = {
            "past_rumination": temporal.past_focus,
            "other_focus": response_style.other_reference_ratio,
            "certainty": response_style.certainty_ratio,
        }

        # Borderline-relevant: emotional expressiveness, absolutism (splitting)
        metrics["borderline_relevant"] = {
            "emotional_intensity": response_style.emotional_expressiveness,
            "black_white_thinking": response_style.absolutism_ratio,
            "self_focus": response_style.self_reference_ratio,
        }

        # Narcissistic-relevant: self-reference, certainty
        metrics["narcissistic_relevant"] = {
            "self_focus": response_style.self_reference_ratio,
            "certainty": response_style.certainty_ratio,
            "vocabulary_sophistication": complexity.vocabulary_sophistication,
        }

        # Avoidant-relevant: hedging, low certainty
        metrics["avoidant_relevant"] = {
            "hedging": response_style.hedging_ratio,
            "uncertainty": 1 - response_style.certainty_ratio,
        }

        # Dependent-relevant: other-reference, hedging
        metrics["dependent_relevant"] = {
            "other_focus": response_style.other_reference_ratio,
            "hedging": response_style.hedging_ratio,
            "self_deprecation": 1 - response_style.certainty_ratio,
        }

        # Antisocial-relevant: deflection, low hedging (confidence)
        metrics["antisocial_relevant"] = {
            "deflection": response_style.deflection_ratio,
            "confidence": 1 - response_style.hedging_ratio,
            "certainty": response_style.certainty_ratio,
        }

        # Obsessive-compulsive-relevant: complexity, certainty (rigidity)
        metrics["obsessive_compulsive_relevant"] = {
            "complexity": complexity.vocabulary_sophistication,
            "rigidity": response_style.absolutism_ratio,
            "detail_orientation": complexity.avg_sentence_length / 20,  # Normalized
        }

        return metrics
