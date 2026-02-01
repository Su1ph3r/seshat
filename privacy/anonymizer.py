"""
Data anonymization for Seshat.

Anonymizes profiles and samples for sharing.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import json
import re


class Anonymizer:
    """
    Anonymize profiles and data for sharing.
    """

    def __init__(self, salt: Optional[str] = None):
        """
        Initialize anonymizer.

        Args:
            salt: Salt for hashing (for consistent anonymization)
        """
        self.salt = salt or "seshat-anon"

    def anonymize_profile(
        self,
        profile,
        remove_samples: bool = False,
        keep_features: bool = True,
    ) -> Dict[str, Any]:
        """
        Anonymize an author profile.

        Args:
            profile: AuthorProfile to anonymize
            remove_samples: Remove sample texts entirely
            keep_features: Keep aggregated features

        Returns:
            Anonymized profile dictionary
        """
        anon_id = self._hash_identifier(profile.profile_id)
        anon_name = f"Anonymous-{anon_id[:8]}"

        result = {
            "profile_id": anon_id,
            "name": anon_name,
            "created_at": self._anonymize_date(profile.created_at),
            "sample_count": len(profile.samples),
            "total_words": profile.total_words,
        }

        if keep_features:
            result["aggregated_features"] = profile.aggregated_features
            result["feature_stats"] = profile.feature_stats

        if not remove_samples:
            result["samples"] = [
                self.anonymize_sample(s) for s in profile.samples
            ]

        return result

    def anonymize_sample(self, sample) -> Dict[str, Any]:
        """
        Anonymize a sample.

        Args:
            sample: Sample to anonymize

        Returns:
            Anonymized sample dictionary
        """
        return {
            "sample_id": self._hash_identifier(str(sample.sample_id) if hasattr(sample, "sample_id") else "unknown"),
            "word_count": sample.word_count,
            "features": sample.features,
            "source_platform": sample.source if hasattr(sample, "source") else None,
        }

    def anonymize_text(
        self,
        text: str,
        redact_names: bool = True,
        redact_locations: bool = True,
        redact_dates: bool = True,
    ) -> str:
        """
        Anonymize text content.

        Args:
            text: Text to anonymize
            redact_names: Redact detected names
            redact_locations: Redact locations
            redact_dates: Redact specific dates

        Returns:
            Anonymized text
        """
        result = text

        if redact_dates:
            date_patterns = [
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                r"\b\d{4}-\d{2}-\d{2}\b",
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
            ]
            for pattern in date_patterns:
                result = re.sub(pattern, "[DATE]", result, flags=re.IGNORECASE)

        return result

    def _hash_identifier(self, identifier: str) -> str:
        """Hash an identifier for anonymization."""
        content = f"{self.salt}:{identifier}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _anonymize_date(self, dt: Optional[datetime]) -> Optional[str]:
        """Anonymize a date (keep only year-month)."""
        if dt is None:
            return None
        return dt.strftime("%Y-%m")

    def export_anonymized(
        self,
        profile,
        output_path: str,
        **kwargs,
    ):
        """
        Export anonymized profile to file.

        Args:
            profile: Profile to anonymize and export
            output_path: Output file path
            **kwargs: Anonymization options
        """
        anon = self.anonymize_profile(profile, **kwargs)

        with open(output_path, "w") as f:
            json.dump(anon, f, indent=2)
