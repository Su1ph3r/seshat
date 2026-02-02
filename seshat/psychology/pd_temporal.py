"""
Temporal analysis for personality disorder indicators.

Analyzes changes in personality disorder indicators across multiple text samples
over time, detecting trends and significant changes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math


@dataclass
class TrendResult:
    """Result of trend analysis for a single disorder or metric."""
    direction: str  # "increasing", "decreasing", "stable", "fluctuating"
    slope: float  # Rate of change per sample
    r_squared: float  # Goodness of fit (0-1)
    start_value: float
    end_value: float
    change_percent: float
    is_significant: bool
    interpretation: str


@dataclass
class ChangePoint:
    """A significant change point in the time series."""
    index: int  # Position in the series
    timestamp: Optional[datetime]
    disorder: str
    before_mean: float
    after_mean: float
    change_magnitude: float
    direction: str  # "increase" or "decrease"


@dataclass
class TemporalAnalysis:
    """Complete temporal analysis across multiple samples."""
    sample_count: int
    time_span: Optional[str]  # Human-readable time span
    disorder_trends: Dict[str, TrendResult]
    cluster_trends: Dict[str, TrendResult]
    change_points: List[ChangePoint]
    stability_score: float  # 0-1, higher = more stable over time
    dominant_pattern: str
    interpretation: str


class PDTemporalAnalyzer:
    """Analyze personality disorder indicators over time."""

    # Minimum samples needed for meaningful trend analysis
    MIN_SAMPLES_FOR_TREND = 3

    # Threshold for considering a change "significant"
    SIGNIFICANT_CHANGE_THRESHOLD = 0.15

    def __init__(self):
        """Initialize the temporal analyzer."""
        self.disorders = [
            "paranoid", "schizoid", "schizotypal",
            "antisocial", "borderline", "histrionic", "narcissistic",
            "avoidant", "dependent", "obsessive_compulsive",
        ]
        self.clusters = {
            "cluster_a": ["paranoid", "schizoid", "schizotypal"],
            "cluster_b": ["antisocial", "borderline", "histrionic", "narcissistic"],
            "cluster_c": ["avoidant", "dependent", "obsessive_compulsive"],
        }

    def analyze_series(
        self,
        results: List[Dict],
        timestamps: Optional[List[datetime]] = None,
    ) -> TemporalAnalysis:
        """
        Analyze a series of personality disorder analysis results over time.

        Args:
            results: List of analysis results from PersonalityDisorderIndicators.analyze()
            timestamps: Optional list of timestamps for each result

        Returns:
            TemporalAnalysis with trend and change point information
        """
        sample_count = len(results)

        if sample_count < 2:
            return self._insufficient_data_result(sample_count)

        # Extract score series for each disorder
        disorder_series = self._extract_disorder_series(results)

        # Calculate cluster averages
        cluster_series = self._calculate_cluster_series(disorder_series)

        # Analyze trends for each disorder
        disorder_trends = {}
        for disorder, scores in disorder_series.items():
            disorder_trends[disorder] = self.detect_trend(scores)

        # Analyze trends for each cluster
        cluster_trends = {}
        for cluster, scores in cluster_series.items():
            cluster_trends[cluster] = self.detect_trend(scores)

        # Find change points
        change_points = self._find_change_points(disorder_series, timestamps)

        # Calculate overall stability
        stability_score = self._calculate_stability(disorder_series)

        # Determine time span
        time_span = None
        if timestamps and len(timestamps) >= 2:
            time_span = self._format_time_span(timestamps[0], timestamps[-1])

        # Determine dominant pattern
        dominant_pattern = self._determine_dominant_pattern(disorder_trends, cluster_trends)

        # Generate interpretation
        interpretation = self._generate_interpretation(
            disorder_trends, cluster_trends, change_points, stability_score
        )

        return TemporalAnalysis(
            sample_count=sample_count,
            time_span=time_span,
            disorder_trends=disorder_trends,
            cluster_trends=cluster_trends,
            change_points=change_points,
            stability_score=stability_score,
            dominant_pattern=dominant_pattern,
            interpretation=interpretation,
        )

    def detect_trend(self, scores: List[float]) -> TrendResult:
        """
        Detect trend in a series of scores.

        Args:
            scores: List of scores over time

        Returns:
            TrendResult with trend analysis
        """
        n = len(scores)

        if n < self.MIN_SAMPLES_FOR_TREND:
            return TrendResult(
                direction="insufficient_data",
                slope=0.0,
                r_squared=0.0,
                start_value=scores[0] if scores else 0.0,
                end_value=scores[-1] if scores else 0.0,
                change_percent=0.0,
                is_significant=False,
                interpretation="Insufficient data for trend analysis",
            )

        # Simple linear regression
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n

        numerator = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Calculate R-squared
        ss_tot = sum((y - y_mean) ** 2 for y in scores)
        ss_res = sum((scores[i] - (y_mean + slope * (i - x_mean))) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Calculate change
        start_value = scores[0]
        end_value = scores[-1]
        change_percent = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0

        # Determine direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Check for fluctuation (low R-squared with non-zero slope)
        if r_squared < 0.3 and abs(slope) >= 0.01:
            direction = "fluctuating"

        # Determine significance
        is_significant = (
            abs(change_percent) >= self.SIGNIFICANT_CHANGE_THRESHOLD * 100 and
            r_squared >= 0.3
        )

        # Generate interpretation
        if direction == "stable":
            interpretation = "Scores remain relatively stable across samples"
        elif direction == "increasing":
            interpretation = f"Scores show upward trend (+{change_percent:.1f}%)"
        elif direction == "decreasing":
            interpretation = f"Scores show downward trend ({change_percent:.1f}%)"
        else:
            interpretation = "Scores show significant fluctuation without clear trend"

        return TrendResult(
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            start_value=start_value,
            end_value=end_value,
            change_percent=change_percent,
            is_significant=is_significant,
            interpretation=interpretation,
        )

    def identify_significant_changes(
        self,
        scores: List[float],
        timestamps: Optional[List[datetime]] = None,
        disorder: str = "unknown",
    ) -> List[ChangePoint]:
        """
        Identify significant change points in a score series.

        Args:
            scores: List of scores over time
            timestamps: Optional timestamps for each score
            disorder: Name of the disorder for the change point record

        Returns:
            List of ChangePoint objects
        """
        if len(scores) < 3:
            return []

        change_points = []

        # Sliding window approach: compare before/after means at each point
        for i in range(1, len(scores) - 1):
            before = scores[:i + 1]
            after = scores[i:]

            before_mean = sum(before) / len(before)
            after_mean = sum(after) / len(after)

            change_magnitude = abs(after_mean - before_mean)

            if change_magnitude >= self.SIGNIFICANT_CHANGE_THRESHOLD:
                change_points.append(ChangePoint(
                    index=i,
                    timestamp=timestamps[i] if timestamps else None,
                    disorder=disorder,
                    before_mean=before_mean,
                    after_mean=after_mean,
                    change_magnitude=change_magnitude,
                    direction="increase" if after_mean > before_mean else "decrease",
                ))

        # Filter to keep only the most significant change point per region
        return self._filter_overlapping_changes(change_points)

    def _extract_disorder_series(self, results: List[Dict]) -> Dict[str, List[float]]:
        """Extract score series for each disorder from results."""
        series = {disorder: [] for disorder in self.disorders}

        for result in results:
            disorders = result.get("disorders", {})
            for disorder in self.disorders:
                score = disorders.get(disorder, {}).get("score", 0.0)
                series[disorder].append(score)

        return series

    def _calculate_cluster_series(
        self,
        disorder_series: Dict[str, List[float]],
    ) -> Dict[str, List[float]]:
        """Calculate cluster average series."""
        sample_count = len(next(iter(disorder_series.values())))
        cluster_series = {}

        for cluster, disorders in self.clusters.items():
            cluster_series[cluster] = []
            for i in range(sample_count):
                cluster_scores = [disorder_series[d][i] for d in disorders]
                cluster_series[cluster].append(sum(cluster_scores) / len(cluster_scores))

        return cluster_series

    def _find_change_points(
        self,
        disorder_series: Dict[str, List[float]],
        timestamps: Optional[List[datetime]],
    ) -> List[ChangePoint]:
        """Find all significant change points across disorders."""
        all_changes = []

        for disorder, scores in disorder_series.items():
            changes = self.identify_significant_changes(scores, timestamps, disorder)
            all_changes.extend(changes)

        # Sort by change magnitude (most significant first)
        all_changes.sort(key=lambda x: x.change_magnitude, reverse=True)

        return all_changes[:10]  # Return top 10 most significant

    def _calculate_stability(self, disorder_series: Dict[str, List[float]]) -> float:
        """Calculate overall stability score (0-1)."""
        if not disorder_series:
            return 0.0

        # Calculate coefficient of variation for each disorder
        cvs = []
        for scores in disorder_series.values():
            if len(scores) < 2:
                continue
            mean = sum(scores) / len(scores)
            if mean == 0:
                cvs.append(0.0)
                continue
            variance = sum((x - mean) ** 2 for x in scores) / len(scores)
            std_dev = math.sqrt(variance)
            cv = std_dev / mean
            cvs.append(cv)

        if not cvs:
            return 0.0

        # Average CV across disorders (higher CV = less stable)
        avg_cv = sum(cvs) / len(cvs)

        # Convert to stability score (invert so higher = more stable)
        # CV of 0 = perfect stability (1.0), CV of 1 = low stability (~0.37)
        stability = math.exp(-avg_cv)

        return min(1.0, max(0.0, stability))

    def _filter_overlapping_changes(self, changes: List[ChangePoint]) -> List[ChangePoint]:
        """Filter to keep only the most significant change per region."""
        if not changes:
            return []

        # Sort by index
        sorted_changes = sorted(changes, key=lambda x: x.index)

        filtered = [sorted_changes[0]]
        for change in sorted_changes[1:]:
            # If this change is close to the last one, keep the larger
            if change.index - filtered[-1].index <= 2:
                if change.change_magnitude > filtered[-1].change_magnitude:
                    filtered[-1] = change
            else:
                filtered.append(change)

        return filtered

    def _format_time_span(self, start: datetime, end: datetime) -> str:
        """Format time span as human-readable string."""
        delta = end - start

        if delta.days > 365:
            years = delta.days / 365
            return f"{years:.1f} years"
        elif delta.days > 30:
            months = delta.days / 30
            return f"{months:.1f} months"
        elif delta.days > 0:
            return f"{delta.days} days"
        elif delta.seconds > 3600:
            hours = delta.seconds / 3600
            return f"{hours:.1f} hours"
        else:
            return f"{delta.seconds // 60} minutes"

    def _determine_dominant_pattern(
        self,
        disorder_trends: Dict[str, TrendResult],
        cluster_trends: Dict[str, TrendResult],
    ) -> str:
        """Determine the dominant pattern across all trends."""
        # Count directions
        direction_counts = {"increasing": 0, "decreasing": 0, "stable": 0, "fluctuating": 0}

        for trend in disorder_trends.values():
            if trend.direction in direction_counts:
                direction_counts[trend.direction] += 1

        # Find dominant direction
        max_count = max(direction_counts.values())
        dominant = [d for d, c in direction_counts.items() if c == max_count][0]

        return dominant

    def _generate_interpretation(
        self,
        disorder_trends: Dict[str, TrendResult],
        cluster_trends: Dict[str, TrendResult],
        change_points: List[ChangePoint],
        stability_score: float,
    ) -> str:
        """Generate overall interpretation of temporal analysis."""
        parts = []

        # Overall stability
        if stability_score >= 0.8:
            parts.append("Indicator patterns are highly stable across samples.")
        elif stability_score >= 0.6:
            parts.append("Indicator patterns show moderate stability.")
        elif stability_score >= 0.4:
            parts.append("Indicator patterns show notable variation across samples.")
        else:
            parts.append("Indicator patterns are highly variable across samples.")

        # Significant trends
        significant_increases = [
            d for d, t in disorder_trends.items()
            if t.is_significant and t.direction == "increasing"
        ]
        significant_decreases = [
            d for d, t in disorder_trends.items()
            if t.is_significant and t.direction == "decreasing"
        ]

        if significant_increases:
            parts.append(f"Significant increases in: {', '.join(significant_increases)}.")
        if significant_decreases:
            parts.append(f"Significant decreases in: {', '.join(significant_decreases)}.")

        # Change points
        if change_points:
            parts.append(f"{len(change_points)} significant change point(s) detected.")

        if not parts:
            parts.append("No significant temporal patterns detected.")

        return " ".join(parts)

    def _insufficient_data_result(self, sample_count: int) -> TemporalAnalysis:
        """Return result for insufficient data."""
        return TemporalAnalysis(
            sample_count=sample_count,
            time_span=None,
            disorder_trends={},
            cluster_trends={},
            change_points=[],
            stability_score=0.0,
            dominant_pattern="unknown",
            interpretation=f"Insufficient data for temporal analysis ({sample_count} samples; minimum 2 required)",
        )

    def compare_samples(
        self,
        result1: Dict,
        result2: Dict,
        timestamp1: Optional[datetime] = None,
        timestamp2: Optional[datetime] = None,
    ) -> Dict:
        """
        Compare two analysis results.

        Args:
            result1: First analysis result
            result2: Second analysis result
            timestamp1: Optional timestamp for first result
            timestamp2: Optional timestamp for second result

        Returns:
            Dictionary with comparison metrics
        """
        disorders1 = result1.get("disorders", {})
        disorders2 = result2.get("disorders", {})

        comparison = {
            "disorders": {},
            "clusters": {},
            "overall_change": 0.0,
            "significant_changes": [],
        }

        # Compare each disorder
        total_change = 0.0
        for disorder in self.disorders:
            score1 = disorders1.get(disorder, {}).get("score", 0.0)
            score2 = disorders2.get(disorder, {}).get("score", 0.0)
            change = score2 - score1

            comparison["disorders"][disorder] = {
                "score1": score1,
                "score2": score2,
                "change": change,
                "change_percent": (change / score1 * 100) if score1 > 0 else 0,
            }

            total_change += abs(change)

            if abs(change) >= self.SIGNIFICANT_CHANGE_THRESHOLD:
                comparison["significant_changes"].append({
                    "disorder": disorder,
                    "change": change,
                    "direction": "increased" if change > 0 else "decreased",
                })

        comparison["overall_change"] = total_change / len(self.disorders)

        # Compare clusters
        for cluster, disorders in self.clusters.items():
            scores1 = [disorders1.get(d, {}).get("score", 0.0) for d in disorders]
            scores2 = [disorders2.get(d, {}).get("score", 0.0) for d in disorders]
            avg1 = sum(scores1) / len(scores1)
            avg2 = sum(scores2) / len(scores2)

            comparison["clusters"][cluster] = {
                "score1": avg1,
                "score2": avg2,
                "change": avg2 - avg1,
            }

        return comparison
