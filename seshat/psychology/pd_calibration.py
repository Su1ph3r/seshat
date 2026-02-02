"""
Calibration layer for personality disorder score normalization.

Provides baseline normalization, genre detection, genre-specific adjustments,
and confidence calibration.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter

from seshat.utils import tokenize_words

from .pd_dictionaries import (
    GENRE_INDICATORS,
    POPULATION_BASELINES,
    GENRE_BASELINE_ADJUSTMENTS,
)


@dataclass
class ConfidenceResult:
    """Result of confidence calibration."""
    level: str  # "very_low", "low", "medium", "high", "very_high"
    score: float  # 0.0 to 1.0
    factors: Dict[str, float]  # Individual contributing factors
    explanation: str


@dataclass
class GenreDetectionResult:
    """Result of genre detection."""
    genre: str  # "formal", "informal", "clinical", "social_media", "neutral"
    confidence: float  # 0.0 to 1.0
    indicator_counts: Dict[str, int]


class PDCalibrationLayer:
    """Score calibration and normalization for personality disorder analysis."""

    CONFIDENCE_THRESHOLDS = {
        "very_low": (0.0, 0.2),
        "low": (0.2, 0.4),
        "medium": (0.4, 0.6),
        "high": (0.6, 0.8),
        "very_high": (0.8, 1.0),
    }

    def __init__(self):
        """Initialize the calibration layer."""
        self.population_baselines = POPULATION_BASELINES
        self.genre_adjustments = GENRE_BASELINE_ADJUSTMENTS
        self.genre_indicators = GENRE_INDICATORS

    def normalize_score(
        self,
        raw_score: float,
        disorder: str,
        method: str = "z_score",
    ) -> float:
        """
        Normalize a raw disorder score against population baselines.

        Args:
            raw_score: The raw score to normalize
            disorder: The disorder name for baseline lookup
            method: Normalization method ("z_score", "percentile", "minmax")

        Returns:
            Normalized score (clamped to 0-1 range)
        """
        if disorder not in self.population_baselines:
            return raw_score

        baseline = self.population_baselines[disorder]
        mean = baseline["mean"]
        std = baseline["std"]

        if method == "z_score":
            # Convert to z-score then transform to 0-1 range
            if std == 0:
                return 0.5  # No variation in baseline
            z = (raw_score - mean) / std
            # Transform z-score to 0-1 using sigmoid-like function
            # z of 0 -> 0.5, z of 2 -> ~0.88, z of -2 -> ~0.12
            normalized = 1 / (1 + pow(2.718281828, -z))

        elif method == "percentile":
            # Simple percentile-like transformation
            # Scores below mean -> 0-0.5, above mean -> 0.5-1
            if raw_score <= mean:
                normalized = (raw_score / mean) * 0.5 if mean > 0 else 0.5
            else:
                # Scale from mean to max (mean + 2*std as approximate max)
                max_expected = mean + 2 * std
                normalized = 0.5 + 0.5 * min(1.0, (raw_score - mean) / (max_expected - mean))

        elif method == "minmax":
            # Simple min-max scaling
            min_expected = max(0, mean - 2 * std)
            max_expected = mean + 2 * std
            if max_expected == min_expected:
                normalized = 0.5
            else:
                normalized = (raw_score - min_expected) / (max_expected - min_expected)

        else:
            normalized = raw_score

        return max(0.0, min(1.0, normalized))

    def normalize_scores(
        self,
        raw_scores: Dict[str, float],
        method: str = "z_score",
    ) -> Dict[str, float]:
        """
        Normalize all disorder scores.

        Args:
            raw_scores: Dictionary of disorder -> raw score
            method: Normalization method

        Returns:
            Dictionary of disorder -> normalized score
        """
        return {
            disorder: self.normalize_score(score, disorder, method)
            for disorder, score in raw_scores.items()
        }

    def detect_genre(self, text: str) -> GenreDetectionResult:
        """
        Detect the genre/register of the text.

        Args:
            text: Input text to analyze

        Returns:
            GenreDetectionResult with detected genre and confidence
        """
        if not text:
            return GenreDetectionResult(
                genre="neutral",
                confidence=0.0,
                indicator_counts={},
            )

        text_lower = text.lower()
        words = set(tokenize_words(text))

        indicator_counts = {}
        max_count = 0
        max_genre = "neutral"

        for genre, indicators in self.genre_indicators.items():
            count = 0
            for indicator in indicators:
                # Check for multi-word indicators
                if ' ' in indicator:
                    if indicator.lower() in text_lower:
                        count += 2  # Multi-word indicators count more
                elif indicator.lower() in words:
                    count += 1
                # Check for special characters (hashtags, @mentions)
                elif indicator in ['#', '@']:
                    count += text.count(indicator)

            indicator_counts[genre] = count
            if count > max_count:
                max_count = count
                max_genre = genre

        # Calculate confidence based on how dominant the top genre is
        total_indicators = sum(indicator_counts.values())
        if total_indicators == 0:
            confidence = 0.0
        else:
            confidence = max_count / total_indicators
            # Adjust confidence based on absolute count
            if max_count < 3:
                confidence *= 0.5  # Low evidence
            elif max_count < 5:
                confidence *= 0.75  # Moderate evidence

        return GenreDetectionResult(
            genre=max_genre if max_count >= 2 else "neutral",
            confidence=min(1.0, confidence),
            indicator_counts=indicator_counts,
        )

    def adjust_for_genre(
        self,
        scores: Dict[str, float],
        genre: str,
    ) -> Dict[str, float]:
        """
        Adjust scores based on detected genre.

        Args:
            scores: Dictionary of disorder -> score
            genre: Detected genre

        Returns:
            Dictionary of disorder -> adjusted score
        """
        if genre not in self.genre_adjustments:
            return scores

        adjustments = self.genre_adjustments[genre]
        adjusted = {}

        for disorder, score in scores.items():
            if disorder in adjustments:
                # Apply multiplicative adjustment
                adjustment_factor = adjustments[disorder]
                # The adjustment factor is applied to how far from baseline the score is
                baseline = self.population_baselines.get(disorder, {}).get("mean", 0.15)
                deviation = score - baseline
                adjusted_deviation = deviation * adjustment_factor
                adjusted[disorder] = max(0.0, min(1.0, baseline + adjusted_deviation))
            else:
                adjusted[disorder] = score

        return adjusted

    def calibrate_confidence(
        self,
        scores: Dict[str, float],
        validation: Dict[str, any],
        word_count: int,
        marker_counts: Optional[Dict[str, int]] = None,
    ) -> ConfidenceResult:
        """
        Calculate overall confidence in the analysis.

        Args:
            scores: Dictionary of disorder -> score
            validation: Validation results (is_consistent, flags, etc.)
            word_count: Number of words in text
            marker_counts: Optional marker counts per disorder

        Returns:
            ConfidenceResult with calibrated confidence level
        """
        factors = {}

        # Factor 1: Text length
        if word_count < 100:
            factors["text_length"] = 0.1
        elif word_count < 200:
            factors["text_length"] = 0.3
        elif word_count < 500:
            factors["text_length"] = 0.5
        elif word_count < 1000:
            factors["text_length"] = 0.7
        elif word_count < 2000:
            factors["text_length"] = 0.85
        else:
            factors["text_length"] = 1.0

        # Factor 2: Consistency
        is_consistent = validation.get("is_consistent", True)
        factors["consistency"] = 1.0 if is_consistent else 0.5

        # Factor 3: Feature coverage
        feature_coverage = validation.get("feature_coverage", 0.0)
        factors["feature_coverage"] = min(1.0, feature_coverage * 2)  # Scale up

        # Factor 4: Score distribution (not all scores at extremes)
        score_values = list(scores.values())
        if score_values:
            score_mean = sum(score_values) / len(score_values)
            # Prefer scores distributed around middle, not all at 0 or 1
            if 0.1 < score_mean < 0.9:
                factors["score_distribution"] = 0.8
            elif score_mean < 0.05 or score_mean > 0.95:
                factors["score_distribution"] = 0.4
            else:
                factors["score_distribution"] = 0.6

        # Factor 5: Marker diversity (if available)
        if marker_counts:
            disorders_with_markers = sum(1 for c in marker_counts.values() if c > 0)
            total_disorders = len(marker_counts)
            factors["marker_diversity"] = disorders_with_markers / total_disorders if total_disorders > 0 else 0
        else:
            factors["marker_diversity"] = 0.5  # Neutral if not available

        # Factor 6: Validation flags
        flags = validation.get("flags", [])
        flag_penalty = len(flags) * 0.1
        factors["validation_flags"] = max(0.0, 1.0 - flag_penalty)

        # Calculate weighted average
        weights = {
            "text_length": 2.0,  # Most important
            "consistency": 1.5,
            "feature_coverage": 1.0,
            "score_distribution": 0.5,
            "marker_diversity": 0.5,
            "validation_flags": 1.0,
        }

        total_weight = sum(weights.values())
        weighted_sum = sum(factors[k] * weights[k] for k in factors)
        confidence_score = weighted_sum / total_weight

        # Determine level
        level = "very_low"
        for lvl, (min_val, max_val) in self.CONFIDENCE_THRESHOLDS.items():
            if min_val <= confidence_score < max_val:
                level = lvl
                break
        if confidence_score >= 1.0:
            level = "very_high"

        # Generate explanation
        explanations = []
        if factors["text_length"] < 0.5:
            explanations.append("Text sample is short")
        if factors["consistency"] < 1.0:
            explanations.append("Some inconsistencies detected")
        if factors["feature_coverage"] < 0.3:
            explanations.append("Limited marker coverage")
        if factors["validation_flags"] < 0.8:
            explanations.append(f"{len(flags)} validation flags raised")

        explanation = "; ".join(explanations) if explanations else "Adequate data for analysis"

        return ConfidenceResult(
            level=level,
            score=confidence_score,
            factors=factors,
            explanation=explanation,
        )

    def calculate_z_score(
        self,
        score: float,
        disorder: str,
    ) -> float:
        """
        Calculate z-score for a disorder score.

        Args:
            score: Raw score
            disorder: Disorder name for baseline lookup

        Returns:
            Z-score (standard deviations from mean)
        """
        if disorder not in self.population_baselines:
            return 0.0

        baseline = self.population_baselines[disorder]
        mean = baseline["mean"]
        std = baseline["std"]

        if std == 0:
            return 0.0

        return (score - mean) / std

    def get_percentile(
        self,
        score: float,
        disorder: str,
    ) -> float:
        """
        Estimate percentile rank for a score (assuming normal distribution).

        Args:
            score: Raw score
            disorder: Disorder name

        Returns:
            Estimated percentile (0-100)
        """
        z = self.calculate_z_score(score, disorder)

        # Approximate percentile from z-score using standard normal CDF
        # Using a polynomial approximation
        if z < -4:
            return 0.01
        if z > 4:
            return 99.99

        # Constants for approximation
        a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
        a4, a5 = -1.453152027, 1.061405429
        p = 0.3275911

        sign = 1 if z >= 0 else -1
        z = abs(z) / pow(2, 0.5)
        t = 1.0 / (1.0 + p * z)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * pow(2.718281828, -z * z)

        percentile = (0.5 * (1.0 + sign * y)) * 100

        return max(0.01, min(99.99, percentile))
