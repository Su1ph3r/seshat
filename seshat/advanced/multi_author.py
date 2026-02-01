"""
Multi-author detection for texts written by multiple people.

Detects documents with multiple authors, ghostwriting, and collaborative writing.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import Counter

from seshat.utils import tokenize_sentences, tokenize_words
from seshat.analyzer import Analyzer


class MultiAuthorDetector:
    """
    Detect if a document was written by multiple authors.

    Uses rolling window style analysis to identify style breaks
    and segment boundaries.
    """

    def __init__(self, analyzer: Optional[Analyzer] = None):
        """
        Initialize multi-author detector.

        Args:
            analyzer: Analyzer instance for feature extraction
        """
        self.analyzer = analyzer or Analyzer()

    def detect(
        self,
        text: str,
        window_size: int = 500,
        step_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Detect multiple authors in a text.

        Args:
            text: Input text to analyze
            window_size: Characters per analysis window
            step_size: Step size between windows

        Returns:
            Multi-author detection results
        """
        if len(text) < window_size * 2:
            return {
                "is_multi_author": False,
                "confidence": "low",
                "reason": "Text too short for multi-author analysis",
                "segments": [],
            }

        window_features = self._extract_window_features(text, window_size, step_size)

        style_distances = self._compute_style_distances(window_features)

        breakpoints = self._detect_breakpoints(style_distances)

        segments = self._create_segments(text, breakpoints, window_size, step_size)

        consistency = self._compute_consistency_score(style_distances)

        is_multi_author = len(breakpoints) > 0 and consistency < 0.7

        if is_multi_author and len(breakpoints) >= 2:
            confidence = "high"
        elif is_multi_author:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "is_multi_author": is_multi_author,
            "confidence": confidence,
            "estimated_authors": len(breakpoints) + 1 if breakpoints else 1,
            "consistency_score": consistency,
            "breakpoints": breakpoints,
            "segments": segments,
            "style_variance": float(np.var(style_distances)) if style_distances else 0,
        }

    def _extract_window_features(
        self,
        text: str,
        window_size: int,
        step_size: int,
    ) -> List[Dict[str, float]]:
        """Extract features for each sliding window."""
        windows = []

        for i in range(0, len(text) - window_size + 1, step_size):
            window_text = text[i:i + window_size]

            analysis = self.analyzer.analyze(window_text)
            features = analysis.get_flat_features()

            features["_position"] = i
            windows.append(features)

        return windows

    def _compute_style_distances(
        self,
        window_features: List[Dict[str, float]],
    ) -> List[float]:
        """Compute style distance between adjacent windows."""
        if len(window_features) < 2:
            return []

        distances = []

        all_keys = set()
        for features in window_features:
            all_keys.update(k for k in features.keys() if not k.startswith("_"))

        for i in range(1, len(window_features)):
            prev_features = window_features[i - 1]
            curr_features = window_features[i]

            squared_diffs = []
            for key in all_keys:
                prev_val = prev_features.get(key, 0)
                curr_val = curr_features.get(key, 0)

                if prev_val != 0 or curr_val != 0:
                    diff = (curr_val - prev_val) ** 2
                    squared_diffs.append(diff)

            distance = np.sqrt(np.mean(squared_diffs)) if squared_diffs else 0
            distances.append(distance)

        return distances

    def _detect_breakpoints(
        self,
        distances: List[float],
        threshold_std: float = 2.0,
    ) -> List[int]:
        """Detect style breakpoints where distance exceeds threshold."""
        if not distances:
            return []

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        threshold = mean_dist + threshold_std * std_dist

        breakpoints = []
        for i, dist in enumerate(distances):
            if dist > threshold:
                if not breakpoints or i - breakpoints[-1] > 3:
                    breakpoints.append(i)

        return breakpoints

    def _create_segments(
        self,
        text: str,
        breakpoints: List[int],
        window_size: int,
        step_size: int,
    ) -> List[Dict[str, Any]]:
        """Create segment information from breakpoints."""
        if not breakpoints:
            return [{
                "segment_id": 0,
                "start": 0,
                "end": len(text),
                "length": len(text),
            }]

        segments = []

        segment_boundaries = [0]
        for bp in breakpoints:
            char_position = bp * step_size + window_size // 2
            segment_boundaries.append(char_position)
        segment_boundaries.append(len(text))

        for i in range(len(segment_boundaries) - 1):
            start = segment_boundaries[i]
            end = segment_boundaries[i + 1]

            segments.append({
                "segment_id": i,
                "start": start,
                "end": end,
                "length": end - start,
            })

        return segments

    def _compute_consistency_score(self, distances: List[float]) -> float:
        """Compute overall style consistency score (0-1)."""
        if not distances:
            return 1.0

        cv = np.std(distances) / (np.mean(distances) + 0.001)

        consistency = 1 / (1 + cv)

        return float(consistency)

    def compare_segments(
        self,
        text: str,
        segment1_range: Tuple[int, int],
        segment2_range: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Compare two segments of text for authorship similarity.

        Args:
            text: Full text
            segment1_range: (start, end) character positions for segment 1
            segment2_range: (start, end) character positions for segment 2

        Returns:
            Comparison results
        """
        segment1 = text[segment1_range[0]:segment1_range[1]]
        segment2 = text[segment2_range[0]:segment2_range[1]]

        analysis1 = self.analyzer.analyze(segment1)
        analysis2 = self.analyzer.analyze(segment2)

        features1 = analysis1.get_flat_features()
        features2 = analysis2.get_flat_features()

        common_keys = set(features1.keys()) & set(features2.keys())

        if not common_keys:
            return {
                "similarity": 0.0,
                "same_author_likely": False,
            }

        vec1 = np.array([features1[k] for k in sorted(common_keys)])
        vec2 = np.array([features2[k] for k in sorted(common_keys)])

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))

        return {
            "similarity": similarity,
            "same_author_likely": similarity > 0.8,
            "feature_count": len(common_keys),
        }

    def detect_ghostwriting(
        self,
        claimed_author_samples: List[str],
        document: str,
    ) -> Dict[str, Any]:
        """
        Detect if a document was ghostwritten (not by claimed author).

        Args:
            claimed_author_samples: Known samples from claimed author
            document: Document to verify

        Returns:
            Ghostwriting detection results
        """
        from seshat.profile import AuthorProfile

        profile = AuthorProfile.create(name="claimed_author")

        for sample in claimed_author_samples:
            try:
                profile.add_sample(sample, source="reference", analyzer=self.analyzer)
            except ValueError:
                continue

        if not profile.samples:
            return {
                "error": "No valid reference samples provided",
            }

        doc_analysis = self.analyzer.analyze(document)
        doc_features = doc_analysis.get_flat_features()

        profile_features = profile.aggregated_features
        profile_stats = profile.feature_statistics

        deviations = []
        for key, value in doc_features.items():
            if key in profile_stats:
                stats = profile_stats[key]
                mean = stats["mean"]
                std = stats["std"]

                if std > 0:
                    z_score = abs(value - mean) / std
                    deviations.append(z_score)

        if deviations:
            avg_deviation = np.mean(deviations)
            max_deviation = np.max(deviations)
        else:
            avg_deviation = 0
            max_deviation = 0

        is_ghostwritten = avg_deviation > 2.0 or max_deviation > 4.0

        if avg_deviation > 3.0:
            confidence = "high"
        elif avg_deviation > 2.0:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "is_ghostwritten": is_ghostwritten,
            "confidence": confidence,
            "average_deviation": float(avg_deviation),
            "max_deviation": float(max_deviation),
            "features_compared": len(deviations),
        }
