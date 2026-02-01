"""
Style drift detector for Seshat.

Detects changes in writing style over time.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class DriftReport:
    """Report of detected style drift."""
    profile_name: str
    analyzed_at: datetime
    drift_score: float
    drift_detected: bool
    threshold: float
    changed_features: List[Dict[str, Any]]
    stable_features: List[str]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "profile_name": self.profile_name,
            "analyzed_at": self.analyzed_at.isoformat(),
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "threshold": self.threshold,
            "changed_features": self.changed_features,
            "stable_features": self.stable_features,
            "recommendation": self.recommendation,
        }


class DriftDetector:
    """
    Detect style drift in author profiles over time.

    Compares recent samples against historical baseline.
    """

    def __init__(
        self,
        drift_threshold: float = 0.15,
        min_samples: int = 5,
    ):
        """
        Initialize drift detector.

        Args:
            drift_threshold: Threshold for drift detection (0-1)
            min_samples: Minimum samples needed for detection
        """
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples

    def detect(
        self,
        profile,
        recent_window: int = 10,
    ) -> DriftReport:
        """
        Detect style drift in a profile.

        Args:
            profile: AuthorProfile to analyze
            recent_window: Number of recent samples to compare

        Returns:
            DriftReport with analysis results
        """
        if len(profile.samples) < self.min_samples:
            return DriftReport(
                profile_name=profile.name,
                analyzed_at=datetime.now(),
                drift_score=0.0,
                drift_detected=False,
                threshold=self.drift_threshold,
                changed_features=[],
                stable_features=[],
                recommendation="Not enough samples for drift detection",
            )

        sorted_samples = sorted(
            profile.samples,
            key=lambda s: s.analyzed_at or datetime.min,
        )

        recent_samples = sorted_samples[-recent_window:]
        baseline_samples = sorted_samples[:-recent_window]

        if len(baseline_samples) < self.min_samples:
            baseline_samples = sorted_samples[:len(sorted_samples) // 2]
            recent_samples = sorted_samples[len(sorted_samples) // 2:]

        baseline_features = self._aggregate_features(baseline_samples)
        recent_features = self._aggregate_features(recent_samples)

        drift_score, changed_features, stable_features = self._compare_features(
            baseline_features,
            recent_features,
        )

        drift_detected = drift_score > self.drift_threshold

        if drift_detected:
            if drift_score > 0.3:
                recommendation = (
                    "Significant style drift detected. Consider investigating "
                    "whether this represents genuine evolution or potential "
                    "account compromise."
                )
            else:
                recommendation = (
                    "Moderate style drift detected. This may be normal evolution "
                    "or context-dependent variation."
                )
        else:
            recommendation = "Style remains consistent with baseline."

        return DriftReport(
            profile_name=profile.name,
            analyzed_at=datetime.now(),
            drift_score=drift_score,
            drift_detected=drift_detected,
            threshold=self.drift_threshold,
            changed_features=changed_features[:10],
            stable_features=stable_features[:10],
            recommendation=recommendation,
        )

    def _aggregate_features(
        self,
        samples: List[Any],
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate features from samples."""
        feature_values: Dict[str, List[float]] = {}

        for sample in samples:
            if not sample.features:
                continue

            for key, value in sample.features.items():
                if isinstance(value, (int, float)):
                    if key not in feature_values:
                        feature_values[key] = []
                    feature_values[key].append(float(value))

        aggregated = {}
        for key, values in feature_values.items():
            if len(values) >= 2:
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

        return aggregated

    def _compare_features(
        self,
        baseline: Dict[str, Dict[str, float]],
        recent: Dict[str, Dict[str, float]],
    ) -> tuple:
        """Compare baseline and recent features."""
        changed_features = []
        stable_features = []
        drift_scores = []

        common_keys = set(baseline.keys()) & set(recent.keys())

        for key in common_keys:
            base_mean = baseline[key]["mean"]
            base_std = baseline[key]["std"]
            recent_mean = recent[key]["mean"]

            if base_std > 0:
                z_score = abs(recent_mean - base_mean) / (base_std + 0.001)
            else:
                z_score = abs(recent_mean - base_mean) / (base_mean + 0.001)

            drift_scores.append(min(z_score / 3, 1.0))

            if z_score > 2.0:
                changed_features.append({
                    "feature": key,
                    "baseline_mean": base_mean,
                    "recent_mean": recent_mean,
                    "z_score": z_score,
                    "change_pct": ((recent_mean - base_mean) / (base_mean + 0.001)) * 100,
                })
            else:
                stable_features.append(key)

        changed_features.sort(key=lambda x: x["z_score"], reverse=True)

        overall_drift = float(np.mean(drift_scores)) if drift_scores else 0.0

        return overall_drift, changed_features, stable_features

    def compare_periods(
        self,
        profile,
        period1_start: datetime,
        period1_end: datetime,
        period2_start: datetime,
        period2_end: datetime,
    ) -> DriftReport:
        """
        Compare style between two time periods.

        Args:
            profile: AuthorProfile to analyze
            period1_start: Start of first period
            period1_end: End of first period
            period2_start: Start of second period
            period2_end: End of second period

        Returns:
            DriftReport with comparison results
        """
        period1_samples = [
            s for s in profile.samples
            if s.analyzed_at and period1_start <= s.analyzed_at <= period1_end
        ]

        period2_samples = [
            s for s in profile.samples
            if s.analyzed_at and period2_start <= s.analyzed_at <= period2_end
        ]

        if len(period1_samples) < self.min_samples or len(period2_samples) < self.min_samples:
            return DriftReport(
                profile_name=profile.name,
                analyzed_at=datetime.now(),
                drift_score=0.0,
                drift_detected=False,
                threshold=self.drift_threshold,
                changed_features=[],
                stable_features=[],
                recommendation="Not enough samples in one or both periods",
            )

        period1_features = self._aggregate_features(period1_samples)
        period2_features = self._aggregate_features(period2_samples)

        drift_score, changed_features, stable_features = self._compare_features(
            period1_features,
            period2_features,
        )

        drift_detected = drift_score > self.drift_threshold

        return DriftReport(
            profile_name=profile.name,
            analyzed_at=datetime.now(),
            drift_score=drift_score,
            drift_detected=drift_detected,
            threshold=self.drift_threshold,
            changed_features=changed_features[:10],
            stable_features=stable_features[:10],
            recommendation=f"Period comparison: {'Drift detected' if drift_detected else 'Consistent style'}",
        )
