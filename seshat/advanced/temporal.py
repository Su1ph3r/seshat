"""
Temporal style analysis for tracking writing style evolution over time.

Analyzes how an author's style changes across different time periods,
useful for long-term surveillance and detecting account takeovers.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np

from seshat.analyzer import Analyzer, AnalysisResult
from seshat.profile import AuthorProfile, Sample


class TemporalAnalyzer:
    """
    Analyze style changes over time.

    Tracks style evolution, detects drift, and identifies stable vs
    variable features.
    """

    def __init__(self, analyzer: Optional[Analyzer] = None):
        """
        Initialize temporal analyzer.

        Args:
            analyzer: Analyzer instance for feature extraction
        """
        self.analyzer = analyzer or Analyzer()

    def analyze_drift(
        self,
        profile: AuthorProfile,
        time_windows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze style drift over time for a profile.

        Args:
            profile: AuthorProfile with timestamped samples
            time_windows: Number of time windows to divide samples into

        Returns:
            Style drift analysis results
        """
        dated_samples = [
            s for s in profile.samples
            if s.timestamp is not None
        ]

        if len(dated_samples) < 2:
            return {
                "has_temporal_data": False,
                "message": "Insufficient timestamped samples for temporal analysis",
            }

        dated_samples.sort(key=lambda s: s.timestamp)

        if time_windows is None:
            time_windows = min(5, len(dated_samples) // 3)
            time_windows = max(2, time_windows)

        window_features = self._compute_window_features(dated_samples, time_windows)

        drift_metrics = self._compute_drift_metrics(window_features)

        stable_features = self._identify_stable_features(window_features)

        variable_features = self._identify_variable_features(window_features)

        return {
            "has_temporal_data": True,
            "time_windows": time_windows,
            "date_range": {
                "start": dated_samples[0].timestamp.isoformat(),
                "end": dated_samples[-1].timestamp.isoformat(),
            },
            "samples_analyzed": len(dated_samples),
            "drift_metrics": drift_metrics,
            "stable_features": stable_features[:20],
            "variable_features": variable_features[:20],
            "window_analysis": window_features,
        }

    def _compute_window_features(
        self,
        samples: List[Sample],
        num_windows: int,
    ) -> List[Dict[str, Any]]:
        """Compute average features for each time window."""
        samples_per_window = len(samples) // num_windows
        if samples_per_window < 1:
            samples_per_window = 1

        windows = []

        for i in range(num_windows):
            start_idx = i * samples_per_window
            end_idx = start_idx + samples_per_window if i < num_windows - 1 else len(samples)

            window_samples = samples[start_idx:end_idx]

            all_features: Dict[str, List[float]] = defaultdict(list)

            for sample in window_samples:
                if sample.analysis:
                    flat = sample.analysis.get_flat_features()
                    for key, value in flat.items():
                        all_features[key].append(value)

            avg_features = {
                key: float(np.mean(values))
                for key, values in all_features.items()
            }

            windows.append({
                "window_index": i,
                "start_date": window_samples[0].timestamp.isoformat() if window_samples[0].timestamp else None,
                "end_date": window_samples[-1].timestamp.isoformat() if window_samples[-1].timestamp else None,
                "sample_count": len(window_samples),
                "features": avg_features,
            })

        return windows

    def _compute_drift_metrics(
        self,
        window_features: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute overall drift metrics."""
        if len(window_features) < 2:
            return {
                "overall_drift": 0.0,
                "drift_direction": "stable",
            }

        first_features = window_features[0]["features"]
        last_features = window_features[-1]["features"]

        common_keys = set(first_features.keys()) & set(last_features.keys())

        if not common_keys:
            return {
                "overall_drift": 0.0,
                "drift_direction": "unknown",
            }

        drifts = []
        for key in common_keys:
            first_val = first_features[key]
            last_val = last_features[key]

            if first_val != 0:
                relative_change = abs(last_val - first_val) / (abs(first_val) + 0.001)
            else:
                relative_change = abs(last_val)

            drifts.append(relative_change)

        overall_drift = float(np.mean(drifts))

        window_distances = []
        for i in range(1, len(window_features)):
            prev_features = window_features[i - 1]["features"]
            curr_features = window_features[i]["features"]

            distances = []
            for key in common_keys:
                prev_val = prev_features.get(key, 0)
                curr_val = curr_features.get(key, 0)
                distances.append(abs(curr_val - prev_val))

            window_distances.append(float(np.mean(distances)))

        if len(window_distances) >= 2:
            if window_distances[-1] > window_distances[0] * 1.5:
                direction = "accelerating"
            elif window_distances[-1] < window_distances[0] * 0.5:
                direction = "stabilizing"
            else:
                direction = "gradual"
        else:
            direction = "stable"

        return {
            "overall_drift": overall_drift,
            "drift_direction": direction,
            "window_distances": window_distances,
        }

    def _identify_stable_features(
        self,
        window_features: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify features that remain stable over time."""
        if not window_features:
            return []

        all_keys = set()
        for window in window_features:
            all_keys.update(window["features"].keys())

        feature_variances = []

        for key in all_keys:
            values = []
            for window in window_features:
                if key in window["features"]:
                    values.append(window["features"][key])

            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / (abs(mean_val) + 0.001)

                feature_variances.append({
                    "feature": key,
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "coefficient_of_variation": float(cv),
                    "stability_score": float(1 / (cv + 0.01)),
                })

        feature_variances.sort(key=lambda x: x["stability_score"], reverse=True)

        return feature_variances

    def _identify_variable_features(
        self,
        window_features: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify features that vary significantly over time."""
        stable = self._identify_stable_features(window_features)

        stable.sort(key=lambda x: x["coefficient_of_variation"], reverse=True)

        return stable

    def detect_anomalies(
        self,
        profile: AuthorProfile,
        new_sample: str,
        threshold: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Detect if a new sample is anomalous compared to profile history.

        Useful for detecting account takeovers or ghostwriting.

        Args:
            profile: AuthorProfile with historical samples
            new_sample: New text to analyze
            threshold: Z-score threshold for anomaly detection

        Returns:
            Anomaly detection results
        """
        analysis = self.analyzer.analyze(new_sample)
        new_features = analysis.get_flat_features()

        anomalies = []

        for key, value in new_features.items():
            if key in profile.feature_statistics:
                stats = profile.feature_statistics[key]
                mean = stats["mean"]
                std = stats["std"]

                if std > 0:
                    z_score = abs(value - mean) / std

                    if z_score > threshold:
                        anomalies.append({
                            "feature": key,
                            "value": value,
                            "profile_mean": mean,
                            "profile_std": std,
                            "z_score": z_score,
                        })

        is_anomalous = len(anomalies) > len(new_features) * 0.1

        return {
            "is_anomalous": is_anomalous,
            "anomaly_count": len(anomalies),
            "total_features": len(new_features),
            "anomaly_ratio": len(anomalies) / len(new_features) if new_features else 0,
            "anomalous_features": sorted(anomalies, key=lambda x: x["z_score"], reverse=True)[:20],
        }

    def compute_style_age(
        self,
        profile: AuthorProfile,
    ) -> Dict[str, Any]:
        """
        Estimate how "mature" an author's style is.

        Newer writers tend to show more style variation over time.

        Args:
            profile: AuthorProfile to analyze

        Returns:
            Style maturity assessment
        """
        drift_analysis = self.analyze_drift(profile)

        if not drift_analysis.get("has_temporal_data"):
            return {
                "has_data": False,
                "message": "Insufficient temporal data",
            }

        overall_drift = drift_analysis["drift_metrics"]["overall_drift"]
        stable_features = len(drift_analysis["stable_features"])

        if overall_drift < 0.1 and stable_features > 50:
            maturity = "mature"
            description = "Highly consistent style with minimal drift"
        elif overall_drift < 0.3:
            maturity = "established"
            description = "Established style with gradual evolution"
        elif overall_drift < 0.5:
            maturity = "developing"
            description = "Style still developing with moderate variation"
        else:
            maturity = "emerging"
            description = "Highly variable style, possibly new writer or significant life changes"

        return {
            "has_data": True,
            "maturity_level": maturity,
            "description": description,
            "drift_score": overall_drift,
            "stable_feature_count": stable_features,
        }
