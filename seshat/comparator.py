"""
Profile comparison and authorship scoring for Seshat.

Implements various statistical methods for comparing stylometric profiles
and calculating attribution confidence scores.
"""

import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine, cityblock, euclidean

from seshat.profile import AuthorProfile
from seshat.analyzer import Analyzer, AnalysisResult


@dataclass
class ComparisonResult:
    """Result of comparing a text against a profile."""

    profile_name: str
    profile_id: str

    overall_score: float
    confidence: str

    burrows_delta: float
    cosine_similarity: float
    manhattan_distance: float
    euclidean_distance: float

    matching_features: List[Dict[str, Any]]
    divergent_features: List[Dict[str, Any]]

    feature_breakdown: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "profile_name": self.profile_name,
            "profile_id": self.profile_id,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "burrows_delta": self.burrows_delta,
            "cosine_similarity": self.cosine_similarity,
            "manhattan_distance": self.manhattan_distance,
            "euclidean_distance": self.euclidean_distance,
            "matching_features": self.matching_features,
            "divergent_features": self.divergent_features,
            "feature_breakdown": self.feature_breakdown,
        }


class Comparator:
    """
    Compares texts against author profiles for authorship attribution.

    Implements multiple distance metrics and combines them for robust
    attribution scoring.
    """

    def __init__(
        self,
        analyzer: Optional[Analyzer] = None,
        delta_weight: float = 0.30,
        cosine_weight: float = 0.30,
        distinctive_weight: float = 0.20,
        consistency_weight: float = 0.20,
    ):
        """
        Initialize the comparator.

        Args:
            analyzer: Analyzer instance for text analysis
            delta_weight: Weight for Burrow's Delta in final score
            cosine_weight: Weight for cosine similarity
            distinctive_weight: Weight for distinctive feature matching
            consistency_weight: Weight for cross-validation consistency
        """
        self.analyzer = analyzer or Analyzer()
        self.delta_weight = delta_weight
        self.cosine_weight = cosine_weight
        self.distinctive_weight = distinctive_weight
        self.consistency_weight = consistency_weight

    def compare(
        self,
        text: str,
        profile: AuthorProfile,
    ) -> ComparisonResult:
        """
        Compare a text against a single author profile.

        Args:
            text: Text to analyze
            profile: Author profile to compare against

        Returns:
            ComparisonResult with detailed comparison metrics
        """
        analysis = self.analyzer.analyze(text)
        text_features = analysis.get_flat_features()

        profile_features = profile.aggregated_features
        profile_stats = profile.feature_statistics

        common_keys = set(text_features.keys()) & set(profile_features.keys())

        if not common_keys:
            return self._empty_result(profile)

        text_vector = np.array([text_features[k] for k in sorted(common_keys)])
        profile_vector = np.array([profile_features[k] for k in sorted(common_keys)])

        burrows_delta = self._calculate_burrows_delta(
            text_features, profile_features, profile_stats, common_keys
        )

        cosine_sim = self._calculate_cosine_similarity(text_vector, profile_vector)

        manhattan = self._calculate_manhattan_distance(text_vector, profile_vector)

        euclidean_dist = self._calculate_euclidean_distance(text_vector, profile_vector)

        matching, divergent = self._find_distinctive_features(
            text_features, profile_features, profile_stats, common_keys
        )

        feature_breakdown = self._calculate_feature_breakdown(
            text_features, profile_features, common_keys
        )

        delta_score = 1 - min(burrows_delta / 2, 1)
        distinctive_score = len(matching) / (len(matching) + len(divergent) + 1)

        overall_score = (
            self.delta_weight * delta_score +
            self.cosine_weight * cosine_sim +
            self.distinctive_weight * distinctive_score +
            self.consistency_weight * cosine_sim
        )

        overall_score = max(0, min(1, overall_score))

        confidence = self._score_to_confidence(overall_score)

        return ComparisonResult(
            profile_name=profile.name,
            profile_id=profile.profile_id,
            overall_score=overall_score,
            confidence=confidence,
            burrows_delta=burrows_delta,
            cosine_similarity=cosine_sim,
            manhattan_distance=manhattan,
            euclidean_distance=euclidean_dist,
            matching_features=matching[:10],
            divergent_features=divergent[:10],
            feature_breakdown=feature_breakdown,
        )

    def compare_multiple(
        self,
        text: str,
        profiles: List[AuthorProfile],
    ) -> List[ComparisonResult]:
        """
        Compare a text against multiple profiles.

        Args:
            text: Text to analyze
            profiles: List of author profiles

        Returns:
            List of ComparisonResults sorted by overall score (descending)
        """
        results = [self.compare(text, profile) for profile in profiles]
        results.sort(key=lambda r: r.overall_score, reverse=True)
        return results

    def compare_texts(
        self,
        text1: str,
        text2: str,
    ) -> Dict[str, Any]:
        """
        Compare two texts directly without profiles.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Comparison metrics between the two texts
        """
        analysis1 = self.analyzer.analyze(text1)
        analysis2 = self.analyzer.analyze(text2)

        features1 = analysis1.get_flat_features()
        features2 = analysis2.get_flat_features()

        common_keys = set(features1.keys()) & set(features2.keys())

        if not common_keys:
            return {
                "similarity": 0.0,
                "cosine_similarity": 0.0,
                "common_features": 0,
            }

        vector1 = np.array([features1[k] for k in sorted(common_keys)])
        vector2 = np.array([features2[k] for k in sorted(common_keys)])

        cosine_sim = self._calculate_cosine_similarity(vector1, vector2)
        manhattan = self._calculate_manhattan_distance(vector1, vector2)
        euclidean_dist = self._calculate_euclidean_distance(vector1, vector2)

        return {
            "similarity": cosine_sim,
            "cosine_similarity": cosine_sim,
            "manhattan_distance": manhattan,
            "euclidean_distance": euclidean_dist,
            "common_features": len(common_keys),
        }

    def _calculate_burrows_delta(
        self,
        text_features: Dict[str, float],
        profile_features: Dict[str, float],
        profile_stats: Dict[str, Dict[str, float]],
        common_keys: set,
    ) -> float:
        """
        Calculate Burrow's Delta distance.

        Delta = (1/n) * Σ |z_text(f) - z_profile(f)|

        Where z-scores normalize feature frequencies.
        """
        if not common_keys:
            return float('inf')

        total_delta = 0.0
        valid_features = 0

        for key in common_keys:
            text_val = text_features[key]
            profile_val = profile_features[key]

            std = profile_stats.get(key, {}).get("std", 1.0)
            if std == 0:
                std = 1.0

            z_text = text_val / std if std != 0 else text_val
            z_profile = profile_val / std if std != 0 else profile_val

            total_delta += abs(z_text - z_profile)
            valid_features += 1

        if valid_features == 0:
            return float('inf')

        return total_delta / valid_features

    def _calculate_cosine_similarity(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray,
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(1 - cosine(vector1, vector2))

    def _calculate_manhattan_distance(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray,
    ) -> float:
        """Calculate Manhattan (L1) distance between two vectors."""
        return float(cityblock(vector1, vector2))

    def _calculate_euclidean_distance(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray,
    ) -> float:
        """Calculate Euclidean (L2) distance between two vectors."""
        return float(euclidean(vector1, vector2))

    def _find_distinctive_features(
        self,
        text_features: Dict[str, float],
        profile_features: Dict[str, float],
        profile_stats: Dict[str, Dict[str, float]],
        common_keys: set,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Find matching and divergent distinctive features.

        Returns:
            Tuple of (matching_features, divergent_features)
        """
        matching = []
        divergent = []

        for key in common_keys:
            text_val = text_features[key]
            profile_val = profile_features[key]
            stats = profile_stats.get(key, {})

            std = stats.get("std", 1.0)
            mean = stats.get("mean", profile_val)

            if std == 0:
                std = 0.1

            z_score = abs(text_val - mean) / std

            feature_info = {
                "feature": key,
                "text_value": text_val,
                "profile_mean": mean,
                "profile_std": std,
                "z_score": z_score,
            }

            if z_score <= 1.5:
                matching.append(feature_info)
            elif z_score > 2.5:
                divergent.append(feature_info)

        matching.sort(key=lambda x: x["z_score"])
        divergent.sort(key=lambda x: x["z_score"], reverse=True)

        return matching, divergent

    def _calculate_feature_breakdown(
        self,
        text_features: Dict[str, float],
        profile_features: Dict[str, float],
        common_keys: set,
    ) -> Dict[str, float]:
        """Calculate per-category similarity scores."""
        categories = {
            "lexical": [],
            "function": [],
            "punctuation": [],
            "formatting": [],
            "ngram": [],
            "syntactic": [],
            "emoji": [],
            "social": [],
        }

        for key in common_keys:
            for cat in categories:
                if key.startswith(cat):
                    text_val = text_features[key]
                    profile_val = profile_features[key]
                    if profile_val != 0:
                        similarity = 1 - min(abs(text_val - profile_val) / (abs(profile_val) + 0.001), 1)
                    else:
                        similarity = 1 if text_val == 0 else 0
                    categories[cat].append(similarity)
                    break

        breakdown = {}
        for cat, similarities in categories.items():
            if similarities:
                breakdown[f"{cat}_similarity"] = sum(similarities) / len(similarities)
            else:
                breakdown[f"{cat}_similarity"] = 0.0

        return breakdown

    def _score_to_confidence(self, score: float) -> str:
        """Convert numeric score to confidence label."""
        if score >= 0.85:
            return "Very High"
        elif score >= 0.70:
            return "High"
        elif score >= 0.55:
            return "Medium"
        elif score >= 0.40:
            return "Low"
        else:
            return "Very Low"

    def _empty_result(self, profile: AuthorProfile) -> ComparisonResult:
        """Return empty result for incompatible comparison."""
        return ComparisonResult(
            profile_name=profile.name,
            profile_id=profile.profile_id,
            overall_score=0.0,
            confidence="Very Low",
            burrows_delta=float('inf'),
            cosine_similarity=0.0,
            manhattan_distance=float('inf'),
            euclidean_distance=float('inf'),
            matching_features=[],
            divergent_features=[],
            feature_breakdown={},
        )


class AttributionEngine:
    """
    High-level authorship attribution engine.

    Wraps Comparator with additional features like confidence intervals
    and attribution explanations.
    """

    def __init__(
        self,
        profiles: Optional[List[AuthorProfile]] = None,
        comparator: Optional[Comparator] = None,
    ):
        """
        Initialize attribution engine.

        Args:
            profiles: List of candidate author profiles
            comparator: Comparator instance
        """
        self.profiles = profiles or []
        self.comparator = comparator or Comparator()

    def add_profile(self, profile: AuthorProfile) -> None:
        """Add a profile to the candidate list."""
        self.profiles.append(profile)

    def attribute(
        self,
        text: str,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform authorship attribution on a text.

        Args:
            text: Text to attribute
            top_n: Number of top candidates to return

        Returns:
            Attribution results with ranked candidates
        """
        if not self.profiles:
            return {
                "error": "No profiles available for attribution",
                "candidates": [],
            }

        results = self.comparator.compare_multiple(text, self.profiles)

        candidates = []
        for result in results[:top_n]:
            candidates.append({
                "profile_name": result.profile_name,
                "score": result.overall_score,
                "confidence": result.confidence,
                "burrows_delta": result.burrows_delta,
                "top_matching": result.matching_features[:5],
                "top_divergent": result.divergent_features[:5],
            })

        best = results[0] if results else None

        explanation = self._generate_explanation(best) if best else ""

        return {
            "best_match": best.profile_name if best else None,
            "best_score": best.overall_score if best else 0.0,
            "confidence": best.confidence if best else "None",
            "candidates": candidates,
            "explanation": explanation,
        }

    def _generate_explanation(self, result: ComparisonResult) -> str:
        """Generate human-readable explanation of attribution."""
        lines = [
            f"Attribution to '{result.profile_name}' with {result.confidence} confidence ({result.overall_score:.1%})",
            "",
            "Key matching features:",
        ]

        for feat in result.matching_features[:5]:
            lines.append(f"  - {feat['feature']}: text={feat['text_value']:.3f}, profile={feat['profile_mean']:.3f}")

        if result.divergent_features:
            lines.append("")
            lines.append("Notable differences:")
            for feat in result.divergent_features[:3]:
                lines.append(f"  - {feat['feature']}: text={feat['text_value']:.3f}, profile={feat['profile_mean']:.3f}")

        return "\n".join(lines)

    def verify_authorship(
        self,
        text: str,
        claimed_author: str,
        threshold: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Verify if a text was written by a claimed author.

        Args:
            text: Text to verify
            claimed_author: Name of claimed author
            threshold: Minimum score to verify authorship

        Returns:
            Verification result
        """
        profile = None
        for p in self.profiles:
            if p.name == claimed_author:
                profile = p
                break

        if profile is None:
            return {
                "verified": False,
                "reason": f"No profile found for '{claimed_author}'",
            }

        result = self.comparator.compare(text, profile)

        verified = result.overall_score >= threshold

        return {
            "verified": verified,
            "score": result.overall_score,
            "confidence": result.confidence,
            "threshold": threshold,
            "reason": (
                f"Score {result.overall_score:.1%} {'meets' if verified else 'below'} "
                f"threshold {threshold:.1%}"
            ),
        }
