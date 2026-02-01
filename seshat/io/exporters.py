"""
Export functionality for profiles and analysis results.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class ProfileExporter:
    """
    Export author profiles in various formats.
    """

    def __init__(self):
        """Initialize exporter."""
        pass

    def export_json(
        self,
        profile: Any,
        path: str,
        include_samples: bool = True,
        indent: int = 2,
    ) -> None:
        """
        Export profile to JSON file.

        Args:
            profile: AuthorProfile object
            path: Output file path
            include_samples: Whether to include raw sample texts
            indent: JSON indentation
        """
        data = profile.to_dict()

        if not include_samples:
            for sample in data.get("samples", []):
                sample["text"] = f"[{len(sample.get('text', ''))} characters]"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)

    def export_csv(
        self,
        profile: Any,
        path: str,
        feature_type: str = "aggregated",
    ) -> None:
        """
        Export profile features to CSV file.

        Args:
            profile: AuthorProfile object
            path: Output file path
            feature_type: "aggregated" or "per_sample"
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if feature_type == "aggregated":
            self._export_aggregated_csv(profile, path)
        else:
            self._export_per_sample_csv(profile, path)

    def _export_aggregated_csv(self, profile: Any, path: Path) -> None:
        """Export aggregated features to CSV."""
        features = profile.aggregated_features
        stats = profile.feature_statistics

        rows = []
        for feature_name in sorted(features.keys()):
            value = features[feature_name]
            feature_stats = stats.get(feature_name, {})

            rows.append({
                "feature": feature_name,
                "mean": value,
                "std": feature_stats.get("std", 0),
                "min": feature_stats.get("min", value),
                "max": feature_stats.get("max", value),
                "sample_count": feature_stats.get("count", 1),
            })

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["feature", "mean", "std", "min", "max", "sample_count"])
            writer.writeheader()
            writer.writerows(rows)

    def _export_per_sample_csv(self, profile: Any, path: Path) -> None:
        """Export per-sample features to CSV."""
        if not profile.samples:
            return

        all_features = set()
        for sample in profile.samples:
            if sample.analysis:
                all_features.update(sample.analysis.get_flat_features().keys())

        fieldnames = ["sample_hash", "source"] + sorted(all_features)

        rows = []
        for sample in profile.samples:
            if sample.analysis:
                flat = sample.analysis.get_flat_features()
                row = {
                    "sample_hash": sample.text_hash[:16],
                    "source": sample.source or "",
                }
                row.update(flat)
                rows.append(row)

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def export_summary(
        self,
        profile: Any,
        path: str,
    ) -> None:
        """
        Export a human-readable profile summary.

        Args:
            profile: AuthorProfile object
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        summary = profile.get_summary()
        distinctive = profile.get_distinctive_features(threshold_std=0.5)

        lines = [
            f"# Author Profile: {summary['name']}",
            f"",
            f"**Profile ID:** {summary['profile_id']}",
            f"**Created:** {summary['created_at']}",
            f"**Last Updated:** {summary['updated_at']}",
            f"",
            f"## Statistics",
            f"",
            f"- **Samples:** {summary['sample_count']}",
            f"- **Total Words:** {summary['total_words']}",
            f"- **Features:** {summary['feature_count']}",
            f"",
            f"## Distinctive Features",
            f"",
            f"Features with high consistency across samples:",
            f"",
        ]

        for feat in distinctive[:20]:
            lines.append(f"- **{feat['feature']}**: {feat['mean']:.4f} (std: {feat['std']:.4f})")

        lines.extend([
            f"",
            f"---",
            f"*Generated by Seshat on {datetime.now().isoformat()}*",
        ])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def export_profile(
    profile: Any,
    path: str,
    format: str = "json",
    **kwargs,
) -> None:
    """
    Export a profile to file.

    Args:
        profile: AuthorProfile object
        path: Output file path
        format: Output format ("json", "csv", "summary")
        **kwargs: Format-specific options
    """
    exporter = ProfileExporter()

    if format == "json":
        exporter.export_json(profile, path, **kwargs)
    elif format == "csv":
        exporter.export_csv(profile, path, **kwargs)
    elif format == "summary":
        exporter.export_summary(profile, path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def export_analysis(
    analysis: Any,
    path: str,
    format: str = "json",
    indent: int = 2,
) -> None:
    """
    Export an analysis result to file.

    Args:
        analysis: AnalysisResult object
        path: Output file path
        format: Output format ("json" or "csv")
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(analysis.to_dict(), f, indent=indent, default=str)

    elif format == "csv":
        flat = analysis.get_flat_features()

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["feature", "value"])
            for key, value in sorted(flat.items()):
                writer.writerow([key, value])

    else:
        raise ValueError(f"Unsupported format: {format}")


def export_comparison(
    results: List[Any],
    path: str,
    format: str = "json",
) -> None:
    """
    Export comparison results to file.

    Args:
        results: List of ComparisonResult objects
        path: Output file path
        format: Output format
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        data = [r.to_dict() for r in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    elif format == "csv":
        if not results:
            return

        fieldnames = [
            "rank", "profile_name", "overall_score", "confidence",
            "burrows_delta", "cosine_similarity",
        ]

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, result in enumerate(results, 1):
                writer.writerow({
                    "rank": i,
                    "profile_name": result.profile_name,
                    "overall_score": f"{result.overall_score:.4f}",
                    "confidence": result.confidence,
                    "burrows_delta": f"{result.burrows_delta:.4f}",
                    "cosine_similarity": f"{result.cosine_similarity:.4f}",
                })

    else:
        raise ValueError(f"Unsupported format: {format}")
