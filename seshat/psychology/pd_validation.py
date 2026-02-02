"""
Validation layer for personality disorder analysis.

Provides cross-disorder discrimination, minimum viable markers checking,
interpersonal circumplex mapping, and validation flag generation.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from .pd_dictionaries import (
    CONTRADICTORY_PAIRS,
    MINIMUM_MARKERS,
    CIRCUMPLEX_COORDINATES,
)


@dataclass
class ValidationResult:
    """Result of discriminant validity checking."""
    is_valid: bool
    contradictions: List[Tuple[str, str, float, float]]  # (disorder1, disorder2, score1, score2)
    warnings: List[str]
    confidence_adjustment: float  # Multiplier for confidence (0.0 to 1.0)


@dataclass
class CircumplexPosition:
    """Position on the interpersonal circumplex."""
    dominance: float  # -1.0 (submissive) to 1.0 (dominant)
    affiliation: float  # -1.0 (hostile) to 1.0 (friendly)
    quadrant: str  # "dominant-hostile", "dominant-friendly", "submissive-hostile", "submissive-friendly"
    angle_degrees: float  # 0-360 degrees
    intensity: float  # Distance from center (0.0 to 1.414)


@dataclass
class ValidationFlags:
    """Collection of validation flags with explanations."""
    flags: List[str] = field(default_factory=list)
    severity: str = "none"  # "none", "low", "medium", "high"
    recommendations: List[str] = field(default_factory=list)


class PDValidationLayer:
    """Cross-validation and discriminant validity for personality disorder analysis."""

    # Score threshold for considering a disorder "elevated"
    ELEVATION_THRESHOLD = 0.4

    # Score threshold for considering contradictory pair problematic
    CONTRADICTION_THRESHOLD = 0.35

    def __init__(self):
        """Initialize the validation layer."""
        self.contradictory_pairs = CONTRADICTORY_PAIRS
        self.minimum_markers = MINIMUM_MARKERS
        self.circumplex_coords = CIRCUMPLEX_COORDINATES

    def check_discriminant_validity(
        self,
        scores: Dict[str, float],
    ) -> ValidationResult:
        """
        Check for contradictory patterns that suggest measurement issues.

        Args:
            scores: Dictionary of disorder -> score

        Returns:
            ValidationResult with validity assessment
        """
        contradictions = []
        warnings = []
        confidence_adjustment = 1.0

        # Check each contradictory pair
        for disorder1, disorder2 in self.contradictory_pairs:
            score1 = scores.get(disorder1, 0.0)
            score2 = scores.get(disorder2, 0.0)

            # Both scores elevated indicates contradiction
            if score1 > self.CONTRADICTION_THRESHOLD and score2 > self.CONTRADICTION_THRESHOLD:
                contradictions.append((disorder1, disorder2, score1, score2))
                warnings.append(
                    f"Contradictory elevation: {disorder1} ({score1:.2f}) and "
                    f"{disorder2} ({score2:.2f}) markers are conceptually opposing"
                )
                # Adjust confidence based on severity
                severity = min(score1, score2) / self.CONTRADICTION_THRESHOLD
                confidence_adjustment *= max(0.5, 1.0 - (severity * 0.2))

        # Check for too many elevated scores (response style artifact)
        elevated_count = sum(1 for s in scores.values() if s > self.ELEVATION_THRESHOLD)
        if elevated_count >= 7:  # More than 70% of disorders elevated
            warnings.append(
                f"Unusually broad elevation ({elevated_count}/10 disorders). "
                "May indicate response style artifact or text characteristics."
            )
            confidence_adjustment *= 0.6
        elif elevated_count >= 5:
            warnings.append(
                f"Multiple elevations ({elevated_count}/10 disorders). "
                "Consider text-specific factors."
            )
            confidence_adjustment *= 0.8

        is_valid = len(contradictions) == 0 and elevated_count < 7

        return ValidationResult(
            is_valid=is_valid,
            contradictions=contradictions,
            warnings=warnings,
            confidence_adjustment=max(0.3, confidence_adjustment),
        )

    def check_minimum_markers(
        self,
        disorder: str,
        markers_found: int,
    ) -> Tuple[bool, str]:
        """
        Check if minimum marker threshold is met for a disorder.

        Args:
            disorder: The disorder name
            markers_found: Number of markers found

        Returns:
            Tuple of (meets_minimum, explanation)
        """
        minimum_required = self.minimum_markers.get(disorder, 3)

        if markers_found >= minimum_required:
            return True, f"Sufficient markers ({markers_found}/{minimum_required})"
        elif markers_found > 0:
            return False, f"Below minimum markers ({markers_found}/{minimum_required}); interpret with caution"
        else:
            return False, "No markers detected"

    def check_all_minimum_markers(
        self,
        marker_counts: Dict[str, int],
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Check minimum markers for all disorders.

        Args:
            marker_counts: Dictionary of disorder -> marker count

        Returns:
            Dictionary of disorder -> (meets_minimum, explanation)
        """
        results = {}
        for disorder, count in marker_counts.items():
            results[disorder] = self.check_minimum_markers(disorder, count)
        return results

    def map_to_circumplex(
        self,
        scores: Dict[str, float],
    ) -> CircumplexPosition:
        """
        Map personality disorder scores to interpersonal circumplex position.

        The circumplex has two dimensions:
        - Dominance: submissive (-1) to dominant (+1)
        - Affiliation: hostile (-1) to friendly (+1)

        Args:
            scores: Dictionary of disorder -> score

        Returns:
            CircumplexPosition with calculated coordinates
        """
        # Weight each disorder's contribution by its score
        total_weight = 0.0
        weighted_dominance = 0.0
        weighted_affiliation = 0.0

        for disorder, score in scores.items():
            if disorder in self.circumplex_coords and score > 0:
                coords = self.circumplex_coords[disorder]
                weight = score
                weighted_dominance += coords["dominance"] * weight
                weighted_affiliation += coords["affiliation"] * weight
                total_weight += weight

        if total_weight == 0:
            dominance = 0.0
            affiliation = 0.0
        else:
            dominance = weighted_dominance / total_weight
            affiliation = weighted_affiliation / total_weight

        # Clamp to valid range
        dominance = max(-1.0, min(1.0, dominance))
        affiliation = max(-1.0, min(1.0, affiliation))

        # Determine quadrant
        if dominance >= 0 and affiliation >= 0:
            quadrant = "dominant-friendly"
        elif dominance >= 0 and affiliation < 0:
            quadrant = "dominant-hostile"
        elif dominance < 0 and affiliation >= 0:
            quadrant = "submissive-friendly"
        else:
            quadrant = "submissive-hostile"

        # Calculate angle (0 = pure dominant, 90 = pure friendly, etc.)
        angle_radians = math.atan2(affiliation, dominance)
        angle_degrees = (math.degrees(angle_radians) + 360) % 360

        # Calculate intensity (distance from center)
        intensity = math.sqrt(dominance ** 2 + affiliation ** 2)

        return CircumplexPosition(
            dominance=dominance,
            affiliation=affiliation,
            quadrant=quadrant,
            angle_degrees=angle_degrees,
            intensity=min(1.414, intensity),  # Max is sqrt(2)
        )

    def generate_validation_flags(
        self,
        scores: Dict[str, float],
        marker_counts: Optional[Dict[str, int]] = None,
        word_count: int = 0,
    ) -> ValidationFlags:
        """
        Generate comprehensive validation flags for the analysis.

        Args:
            scores: Dictionary of disorder -> score
            marker_counts: Optional dictionary of disorder -> marker count
            word_count: Number of words in analyzed text

        Returns:
            ValidationFlags with all detected issues
        """
        flags = []
        recommendations = []

        # Check text length
        if word_count < 100:
            flags.append("Very short text sample (<100 words)")
            recommendations.append("Obtain longer text sample for reliable analysis")
        elif word_count < 200:
            flags.append("Short text sample (<200 words)")
            recommendations.append("Consider obtaining additional text")
        elif word_count < 500:
            flags.append("Text below recommended minimum (500 words)")

        # Check discriminant validity
        validity = self.check_discriminant_validity(scores)
        flags.extend(validity.warnings)
        if not validity.is_valid:
            recommendations.append("Review for response style artifacts or text-specific factors")

        # Check minimum markers if available
        if marker_counts:
            for disorder, count in marker_counts.items():
                if scores.get(disorder, 0) > self.ELEVATION_THRESHOLD:
                    meets_min, explanation = self.check_minimum_markers(disorder, count)
                    if not meets_min:
                        flags.append(f"{disorder}: {explanation}")

        # Check for extreme scores
        very_high_scores = [d for d, s in scores.items() if s > 0.8]
        if very_high_scores:
            flags.append(f"Very high scores ({', '.join(very_high_scores)}); verify against clinical context")
            recommendations.append("Consider whether text context explains elevated markers")

        # Check for uniform low scores
        if all(s < 0.1 for s in scores.values()):
            flags.append("All scores very low; may indicate limited relevant content")

        # Determine severity
        if len(flags) >= 4:
            severity = "high"
        elif len(flags) >= 2:
            severity = "medium"
        elif len(flags) >= 1:
            severity = "low"
        else:
            severity = "none"

        return ValidationFlags(
            flags=flags,
            severity=severity,
            recommendations=recommendations,
        )

    def get_cluster_consistency(
        self,
        scores: Dict[str, float],
    ) -> Dict[str, Dict[str, any]]:
        """
        Analyze consistency within DSM-5 clusters.

        Args:
            scores: Dictionary of disorder -> score

        Returns:
            Dictionary with cluster consistency analysis
        """
        clusters = {
            "cluster_a": ["paranoid", "schizoid", "schizotypal"],
            "cluster_b": ["antisocial", "borderline", "histrionic", "narcissistic"],
            "cluster_c": ["avoidant", "dependent", "obsessive_compulsive"],
        }

        results = {}
        for cluster_name, disorders in clusters.items():
            cluster_scores = [scores.get(d, 0.0) for d in disorders]

            if not cluster_scores:
                continue

            mean_score = sum(cluster_scores) / len(cluster_scores)
            variance = sum((s - mean_score) ** 2 for s in cluster_scores) / len(cluster_scores)
            std_dev = math.sqrt(variance)

            # Coefficient of variation (normalized measure of spread)
            cv = std_dev / mean_score if mean_score > 0 else 0

            # High CV within a cluster may indicate inconsistent pattern
            is_consistent = cv < 1.0  # CV > 1 means high variation

            results[cluster_name] = {
                "mean_score": mean_score,
                "std_dev": std_dev,
                "coefficient_of_variation": cv,
                "is_consistent": is_consistent,
                "highest": max(disorders, key=lambda d: scores.get(d, 0)),
                "lowest": min(disorders, key=lambda d: scores.get(d, 0)),
            }

        return results

    def calculate_profile_clarity(
        self,
        scores: Dict[str, float],
    ) -> Dict[str, any]:
        """
        Calculate how clear/distinct the profile is.

        A clear profile has 1-2 elevated disorders with others low.
        An unclear profile has many moderate scores.

        Args:
            scores: Dictionary of disorder -> score

        Returns:
            Dictionary with clarity metrics
        """
        score_values = sorted(scores.values(), reverse=True)

        if not score_values:
            return {
                "clarity_score": 0.0,
                "clarity_label": "unclear",
                "peak_count": 0,
                "elevation_spread": 0.0,
            }

        # Count "peaks" (scores above elevation threshold)
        peaks = sum(1 for s in score_values if s > self.ELEVATION_THRESHOLD)

        # Calculate spread between highest and others
        if len(score_values) >= 2:
            top_to_second = score_values[0] - score_values[1]
            top_to_mean = score_values[0] - (sum(score_values[1:]) / len(score_values[1:]))
        else:
            top_to_second = 0
            top_to_mean = 0

        # Clarity score: higher when there's a clear peak
        if peaks == 0:
            clarity_score = 0.3  # Nothing elevated
        elif peaks == 1:
            clarity_score = 0.9  # Single clear peak
        elif peaks == 2:
            clarity_score = 0.7  # Two peaks can be meaningful
        elif peaks <= 3:
            clarity_score = 0.5  # Multiple peaks reduce clarity
        else:
            clarity_score = 0.2  # Too many peaks

        # Adjust by separation from others
        clarity_score = min(1.0, clarity_score + (top_to_second * 0.3))

        if clarity_score >= 0.7:
            clarity_label = "clear"
        elif clarity_score >= 0.4:
            clarity_label = "moderate"
        else:
            clarity_label = "unclear"

        return {
            "clarity_score": clarity_score,
            "clarity_label": clarity_label,
            "peak_count": peaks,
            "elevation_spread": top_to_second,
            "top_to_mean_diff": top_to_mean,
        }
