"""
Cross-platform consistency analysis.

Analyzes writing style consistency across different platforms and contexts.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from seshat.analyzer import Analyzer
from seshat.profile import AuthorProfile


class CrossPlatformAnalyzer:
    """
    Analyze style consistency across different platforms.

    Identifies which features remain stable across platforms
    (useful for linking accounts) and which vary with context.
    """

    PLATFORM_STABLE_FEATURES = [
        "lexical_type_token_ratio",
        "lexical_avg_word_length",
        "function_first_person_singular_ratio",
        "function_i_ratio",
        "punctuation_comma_per_1k",
        "punctuation_terminal_question_ratio",
        "formatting_sentence_initial_cap_consistency",
    ]

    PLATFORM_VARIABLE_FEATURES = [
        "emoji_emoji_per_100_words",
        "social_hashtag_per_100_words",
        "social_mention_per_100_words",
        "formatting_avg_paragraph_length_sentences",
        "lexical_formal_ratio",
        "lexical_slang_ratio",
    ]

    def __init__(self, analyzer: Optional[Analyzer] = None):
        """
        Initialize cross-platform analyzer.

        Args:
            analyzer: Analyzer instance
        """
        self.analyzer = analyzer or Analyzer()

    def compare_platforms(
        self,
        samples_by_platform: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """
        Compare writing style across multiple platforms.

        Args:
            samples_by_platform: Dictionary mapping platform name to list of samples

        Returns:
            Cross-platform comparison results
        """
        if len(samples_by_platform) < 2:
            return {
                "error": "Need at least 2 platforms for comparison",
            }

        platform_profiles = {}

        for platform, samples in samples_by_platform.items():
            profile = AuthorProfile.create(name=f"{platform}_profile")
            for sample in samples:
                try:
                    profile.add_sample(sample, source=platform, analyzer=self.analyzer)
                except (ValueError, Exception):
                    continue

            if profile.samples:
                platform_profiles[platform] = profile

        if len(platform_profiles) < 2:
            return {
                "error": "Not enough valid samples across platforms",
            }

        consistency = self._compute_cross_platform_consistency(platform_profiles)

        stable_features = self._identify_stable_features(platform_profiles)

        variable_features = self._identify_variable_features(platform_profiles)

        pairwise = self._compute_pairwise_similarity(platform_profiles)

        same_author = self._assess_same_author(consistency, pairwise)

        return {
            "platforms_analyzed": list(platform_profiles.keys()),
            "overall_consistency": consistency,
            "same_author_assessment": same_author,
            "stable_features": stable_features[:15],
            "variable_features": variable_features[:15],
            "pairwise_similarity": pairwise,
        }

    def _compute_cross_platform_consistency(
        self,
        platform_profiles: Dict[str, AuthorProfile],
    ) -> Dict[str, Any]:
        """Compute overall cross-platform consistency."""
        all_features: Dict[str, List[float]] = {}

        for platform, profile in platform_profiles.items():
            for key, value in profile.aggregated_features.items():
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(value)

        feature_consistencies = []
        for key, values in all_features.items():
            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / (abs(mean_val) + 0.001)
                consistency = 1 / (1 + cv)
                feature_consistencies.append(consistency)

        overall = float(np.mean(feature_consistencies)) if feature_consistencies else 0

        return {
            "overall_score": overall,
            "interpretation": (
                "High consistency" if overall > 0.7
                else "Moderate consistency" if overall > 0.5
                else "Low consistency"
            ),
        }

    def _identify_stable_features(
        self,
        platform_profiles: Dict[str, AuthorProfile],
    ) -> List[Dict[str, Any]]:
        """Identify features that remain stable across platforms."""
        all_features: Dict[str, List[float]] = {}

        for platform, profile in platform_profiles.items():
            for key, value in profile.aggregated_features.items():
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(value)

        stable = []
        for key, values in all_features.items():
            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / (abs(mean_val) + 0.001)

                if cv < 0.3:
                    stable.append({
                        "feature": key,
                        "mean": float(mean_val),
                        "std": float(std_val),
                        "coefficient_of_variation": float(cv),
                        "consistency_score": float(1 / (1 + cv)),
                    })

        stable.sort(key=lambda x: x["consistency_score"], reverse=True)
        return stable

    def _identify_variable_features(
        self,
        platform_profiles: Dict[str, AuthorProfile],
    ) -> List[Dict[str, Any]]:
        """Identify features that vary significantly across platforms."""
        all_features: Dict[str, List[float]] = {}

        for platform, profile in platform_profiles.items():
            for key, value in profile.aggregated_features.items():
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(value)

        variable = []
        for key, values in all_features.items():
            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / (abs(mean_val) + 0.001)

                if cv > 0.5:
                    variable.append({
                        "feature": key,
                        "mean": float(mean_val),
                        "std": float(std_val),
                        "coefficient_of_variation": float(cv),
                        "variability_score": float(cv),
                    })

        variable.sort(key=lambda x: x["variability_score"], reverse=True)
        return variable

    def _compute_pairwise_similarity(
        self,
        platform_profiles: Dict[str, AuthorProfile],
    ) -> Dict[str, float]:
        """Compute pairwise similarity between platforms."""
        platforms = list(platform_profiles.keys())
        similarities = {}

        for i, p1 in enumerate(platforms):
            for p2 in platforms[i + 1:]:
                profile1 = platform_profiles[p1]
                profile2 = platform_profiles[p2]

                common_keys = (
                    set(profile1.aggregated_features.keys()) &
                    set(profile2.aggregated_features.keys())
                )

                if not common_keys:
                    similarity = 0.0
                else:
                    vec1 = np.array([profile1.aggregated_features[k] for k in sorted(common_keys)])
                    vec2 = np.array([profile2.aggregated_features[k] for k in sorted(common_keys)])

                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)

                    if norm1 == 0 or norm2 == 0:
                        similarity = 0.0
                    else:
                        similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))

                similarities[f"{p1}_vs_{p2}"] = similarity

        return similarities

    def _assess_same_author(
        self,
        consistency: Dict[str, Any],
        pairwise: Dict[str, float],
    ) -> Dict[str, Any]:
        """Assess likelihood that all samples are from the same author."""
        overall_consistency = consistency.get("overall_score", 0)

        avg_pairwise = (
            np.mean(list(pairwise.values()))
            if pairwise else 0
        )

        if overall_consistency > 0.7 and avg_pairwise > 0.8:
            likelihood = "likely same author"
            confidence = "high"
        elif overall_consistency > 0.5 and avg_pairwise > 0.6:
            likelihood = "possibly same author"
            confidence = "medium"
        elif overall_consistency > 0.3:
            likelihood = "uncertain"
            confidence = "low"
        else:
            likelihood = "possibly different authors"
            confidence = "medium"

        return {
            "assessment": likelihood,
            "confidence": confidence,
            "overall_consistency": overall_consistency,
            "average_pairwise_similarity": float(avg_pairwise),
        }

    def link_anonymous_account(
        self,
        known_profile: AuthorProfile,
        anonymous_samples: List[str],
        platform: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Assess if an anonymous account belongs to a known author.

        Args:
            known_profile: Profile of known author
            anonymous_samples: Samples from anonymous account
            platform: Platform of anonymous account

        Returns:
            Account linking assessment
        """
        anon_profile = AuthorProfile.create(name="anonymous")

        for sample in anonymous_samples:
            try:
                anon_profile.add_sample(sample, source=platform, analyzer=self.analyzer)
            except (ValueError, Exception):
                continue

        if not anon_profile.samples:
            return {
                "error": "No valid anonymous samples",
            }

        result = self.compare_platforms({
            "known": [s.text for s in known_profile.samples],
            "anonymous": [s.text for s in anon_profile.samples],
        })

        pairwise_sim = result.get("pairwise_similarity", {}).get("known_vs_anonymous", 0)
        consistency = result.get("overall_consistency", {}).get("overall_score", 0)

        if pairwise_sim > 0.85:
            match_assessment = "strong match"
            confidence = "high"
        elif pairwise_sim > 0.7:
            match_assessment = "likely match"
            confidence = "medium"
        elif pairwise_sim > 0.5:
            match_assessment = "possible match"
            confidence = "low"
        else:
            match_assessment = "unlikely match"
            confidence = "medium"

        return {
            "match_assessment": match_assessment,
            "confidence": confidence,
            "similarity_score": pairwise_sim,
            "stable_matching_features": result.get("stable_features", [])[:10],
            "detailed_comparison": result,
        }
