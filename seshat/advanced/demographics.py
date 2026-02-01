"""
Demographic estimation from writing style.

Estimates age, education level, and regional background from linguistic patterns.

IMPORTANT: These are probabilistic estimates based on linguistic correlations,
not definitive determinations. They should be used with appropriate caveats.
"""

from typing import Dict, List, Any, Optional
from collections import Counter
import re

from seshat.utils import tokenize_words, tokenize_sentences


class DemographicEstimator:
    """
    Estimate demographic characteristics from writing style.

    Provides probabilistic estimates for:
    - Age group
    - Education level
    - Regional/dialect indicators
    """

    def __init__(self):
        """Initialize demographic estimator."""
        self.slang_by_generation = self._load_generational_slang()
        self.regional_markers = self._load_regional_markers()

    def _load_generational_slang(self) -> Dict[str, Dict[str, List[str]]]:
        """Load generational vocabulary markers."""
        return {
            "gen_z": {
                "slang": [
                    "bussin", "cap", "nocap", "fr", "frfr", "bet", "slay",
                    "sus", "lowkey", "highkey", "deadass", "periodt", "ong",
                    "no cap", "mid", "hits different", "vibe check", "understood the assignment",
                    "rent free", "main character", "its giving", "ate", "iykyk",
                ],
                "emoji_heavy": True,
                "abbreviation_heavy": True,
            },
            "millennial": {
                "slang": [
                    "adulting", "literally", "basic", "fomo", "yolo", "bae",
                    "on fleek", "squad", "goals", "savage", "salty", "extra",
                    "thirsty", "ship", "stan", "tea", "mood", "woke",
                ],
                "emoji_heavy": True,
                "abbreviation_heavy": True,
            },
            "gen_x": {
                "slang": [
                    "rad", "gnarly", "bogus", "dude", "whatever", "chill",
                    "phat", "da bomb", "all that", "talk to the hand",
                ],
                "emoji_heavy": False,
                "abbreviation_heavy": False,
            },
            "boomer": {
                "markers": [
                    "back in my day", "kids these days", "nowadays",
                    "the good old days", "when i was young",
                ],
                "formal_tendency": True,
            },
        }

    def _load_regional_markers(self) -> Dict[str, Dict[str, Any]]:
        """Load regional/dialect markers."""
        return {
            "american": {
                "spelling": ["color", "favor", "honor", "organize", "realize", "center", "theater"],
                "vocabulary": ["apartment", "truck", "fall", "pants", "cookie", "elevator"],
            },
            "british": {
                "spelling": ["colour", "favour", "honour", "organise", "realise", "centre", "theatre"],
                "vocabulary": ["flat", "lorry", "autumn", "trousers", "biscuit", "lift"],
            },
            "australian": {
                "vocabulary": ["arvo", "brekkie", "servo", "bottle-o", "maccas", "barbie"],
                "slang": ["mate", "no worries", "heaps", "reckon", "keen"],
            },
            "canadian": {
                "vocabulary": ["eh", "toque", "loonie", "toonie", "double-double"],
                "spelling_mixed": True,
            },
        }

    def estimate(self, text: str) -> Dict[str, Any]:
        """
        Estimate demographic characteristics from text.

        Args:
            text: Input text to analyze

        Returns:
            Demographic estimation results
        """
        if not text or len(text) < 100:
            return {
                "disclaimer": "Estimates are probabilistic and should not be considered definitive",
                "age_estimate": self._empty_age_estimate(),
                "education_estimate": self._empty_education_estimate(),
                "regional_estimate": self._empty_regional_estimate(),
            }

        words = tokenize_words(text)
        sentences = tokenize_sentences(text)

        age_estimate = self._estimate_age(text, words)
        education_estimate = self._estimate_education(text, words, sentences)
        regional_estimate = self._estimate_region(text, words)

        return {
            "disclaimer": "Estimates are probabilistic and should not be considered definitive",
            "age_estimate": age_estimate,
            "education_estimate": education_estimate,
            "regional_estimate": regional_estimate,
        }

    def _empty_age_estimate(self) -> Dict[str, Any]:
        """Return empty age estimate."""
        return {
            "estimated_generation": "unknown",
            "confidence": "low",
            "indicators": [],
        }

    def _empty_education_estimate(self) -> Dict[str, Any]:
        """Return empty education estimate."""
        return {
            "estimated_level": "unknown",
            "confidence": "low",
            "indicators": [],
        }

    def _empty_regional_estimate(self) -> Dict[str, Any]:
        """Return empty regional estimate."""
        return {
            "estimated_region": "unknown",
            "confidence": "low",
            "indicators": [],
        }

    def _estimate_age(
        self, text: str, words: List[str]
    ) -> Dict[str, Any]:
        """Estimate age group from generational markers."""
        text_lower = text.lower()
        word_set = set(words)

        generation_scores = {}
        indicators = {}

        for gen, markers in self.slang_by_generation.items():
            score = 0
            found_markers = []

            slang_list = markers.get("slang", []) + markers.get("markers", [])
            for slang in slang_list:
                if slang in text_lower or slang in word_set:
                    score += 1
                    found_markers.append(slang)

            generation_scores[gen] = score
            if found_markers:
                indicators[gen] = found_markers

        if sum(generation_scores.values()) == 0:
            return {
                "estimated_generation": "unknown",
                "confidence": "low",
                "indicators": [],
                "scores": generation_scores,
            }

        top_gen = max(generation_scores.items(), key=lambda x: x[1])

        if top_gen[1] >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        gen_to_age = {
            "gen_z": "18-27",
            "millennial": "28-43",
            "gen_x": "44-59",
            "boomer": "60+",
        }

        return {
            "estimated_generation": top_gen[0],
            "estimated_age_range": gen_to_age.get(top_gen[0], "unknown"),
            "confidence": confidence,
            "indicators": indicators.get(top_gen[0], []),
            "scores": generation_scores,
        }

    def _estimate_education(
        self,
        text: str,
        words: List[str],
        sentences: List[str],
    ) -> Dict[str, Any]:
        """Estimate education level from linguistic complexity."""
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0

        unique_words = len(set(words))
        total_words = len(words)
        ttr = unique_words / total_words if total_words > 0 else 0

        avg_sentence_length = total_words / len(sentences) if sentences else 0

        complex_words = [w for w in words if len(w) > 8]
        complex_ratio = len(complex_words) / len(words) if words else 0

        score = 0
        indicators = []

        if avg_word_length > 5.5:
            score += 2
            indicators.append("High average word length")
        elif avg_word_length > 4.5:
            score += 1

        if ttr > 0.6:
            score += 2
            indicators.append("High vocabulary diversity")
        elif ttr > 0.4:
            score += 1

        if avg_sentence_length > 20:
            score += 2
            indicators.append("Long average sentence length")
        elif avg_sentence_length > 15:
            score += 1

        if complex_ratio > 0.15:
            score += 2
            indicators.append("High proportion of complex words")
        elif complex_ratio > 0.08:
            score += 1

        if score >= 7:
            level = "graduate"
            confidence = "medium"
        elif score >= 5:
            level = "college"
            confidence = "medium"
        elif score >= 3:
            level = "high_school"
            confidence = "low"
        else:
            level = "basic"
            confidence = "low"

        return {
            "estimated_level": level,
            "confidence": confidence,
            "indicators": indicators,
            "metrics": {
                "avg_word_length": avg_word_length,
                "vocabulary_diversity": ttr,
                "avg_sentence_length": avg_sentence_length,
                "complex_word_ratio": complex_ratio,
            },
        }

    def _estimate_region(
        self, text: str, words: List[str]
    ) -> Dict[str, Any]:
        """Estimate regional background from dialect markers."""
        text_lower = text.lower()
        word_set = set(words)

        region_scores = {}
        indicators = {}

        for region, markers in self.regional_markers.items():
            score = 0
            found_markers = []

            for spelling in markers.get("spelling", []):
                if spelling in word_set:
                    score += 2
                    found_markers.append(f"Spelling: {spelling}")

            for vocab in markers.get("vocabulary", []):
                if vocab in text_lower:
                    score += 1
                    found_markers.append(f"Vocabulary: {vocab}")

            for slang in markers.get("slang", []):
                if slang in text_lower:
                    score += 1
                    found_markers.append(f"Slang: {slang}")

            region_scores[region] = score
            if found_markers:
                indicators[region] = found_markers

        if sum(region_scores.values()) == 0:
            return {
                "estimated_region": "unknown",
                "confidence": "low",
                "indicators": [],
                "scores": region_scores,
            }

        top_region = max(region_scores.items(), key=lambda x: x[1])

        if top_region[1] >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "estimated_region": top_region[0],
            "confidence": confidence,
            "indicators": indicators.get(top_region[0], []),
            "scores": region_scores,
        }
