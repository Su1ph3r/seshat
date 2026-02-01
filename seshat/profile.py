"""
Author profile management for Seshat.

Handles creation, storage, and manipulation of stylometric author profiles.
"""

import json
import hashlib
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

from seshat.analyzer import Analyzer, AnalysisResult


@dataclass
class Sample:
    """Represents a single writing sample."""

    text: str
    text_hash: str = ""
    word_count: int = 0
    features: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    source_url: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis: Optional[AnalysisResult] = None

    def __post_init__(self):
        """Compute hash if not provided."""
        if not self.text_hash:
            self.text_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "text_hash": self.text_hash,
            "word_count": self.word_count,
            "features": self.features,
            "source": self.source,
            "source_url": self.source_url,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
            "analysis": self.analysis.to_dict() if self.analysis else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sample":
        """Create from dictionary."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])

        analysis = None
        if data.get("analysis"):
            analysis = AnalysisResult.from_dict(data["analysis"])

        return cls(
            text=data["text"],
            text_hash=data["text_hash"],
            word_count=data.get("word_count", 0),
            features=data.get("features", {}),
            source=data.get("source"),
            source_url=data.get("source_url"),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
            analysis=analysis,
        )

    @classmethod
    def create(
        cls,
        text: str,
        source: Optional[str] = None,
        source_url: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Sample":
        """Create a new sample with computed hash."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(
            text=text,
            text_hash=text_hash,
            source=source,
            source_url=source_url,
            timestamp=timestamp or datetime.now(),
            metadata=metadata or {},
        )


@dataclass
class AuthorProfile:
    """
    Represents an author's stylometric profile.

    Built from multiple writing samples, aggregating features into
    a characteristic "fingerprint".
    """

    name: str
    profile_id: str
    created_at: datetime
    updated_at: datetime

    samples: List[Sample] = field(default_factory=list)

    aggregated_features: Dict[str, Any] = field(default_factory=dict)
    feature_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure profile_id is set."""
        if not self.profile_id:
            self.profile_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique profile ID."""
        unique_string = f"{self.name}_{datetime.now().isoformat()}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]

    @classmethod
    def create(
        cls,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AuthorProfile":
        """Create a new empty author profile."""
        now = datetime.now()
        profile = cls(
            name=name,
            profile_id="",
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        profile.profile_id = profile._generate_id()
        return profile

    def add_sample(
        self,
        text: str,
        source: Optional[str] = None,
        source_url: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        analyzer: Optional[Analyzer] = None,
        min_words: int = 5,
    ) -> Sample:
        """
        Add a writing sample to the profile.

        Args:
            text: The text content
            source: Source identifier (e.g., "twitter", "email")
            source_url: URL where text was found
            timestamp: When the text was written
            metadata: Additional metadata
            analyzer: Analyzer instance (creates one if not provided)
            min_words: Minimum number of words required

        Returns:
            The created Sample object

        Raises:
            ValueError: If text is too short
        """
        from seshat.utils import tokenize_words
        words = tokenize_words(text)
        if len(words) < min_words:
            raise ValueError(f"Sample too short: {len(words)} words (minimum: {min_words})")

        sample = Sample.create(
            text=text,
            source=source,
            source_url=source_url,
            timestamp=timestamp,
            metadata=metadata,
        )

        if self._is_duplicate(sample.text_hash):
            raise ValueError(f"Duplicate sample: {sample.text_hash[:8]}...")

        if analyzer is None:
            analyzer = Analyzer()

        sample.analysis = analyzer.analyze(text, metadata={"source": source})

        self.samples.append(sample)
        self.updated_at = datetime.now()

        self._update_aggregated_features()

        return sample

    def add_samples(
        self,
        texts: List[str],
        source: Optional[str] = None,
        analyzer: Optional[Analyzer] = None,
    ) -> Tuple[List[Sample], List[Dict[str, Any]]]:
        """
        Add multiple samples at once.

        Args:
            texts: List of text contents
            source: Source identifier for all texts
            analyzer: Analyzer instance

        Returns:
            Tuple of (successful_samples, failed_samples_info)
        """
        if analyzer is None:
            analyzer = Analyzer()

        samples = []
        failures = []
        for i, text in enumerate(texts):
            try:
                sample = self.add_sample(text, source=source, analyzer=analyzer)
                samples.append(sample)
            except ValueError as e:
                failure_info = {
                    "index": i,
                    "reason": str(e),
                    "text_preview": text[:100] + "..." if len(text) > 100 else text
                }
                failures.append(failure_info)
                logger.debug(f"Skipping sample {i}: {e}")

        if failures:
            warnings.warn(
                f"Failed to add {len(failures)} of {len(texts)} samples. "
                f"Use logging.DEBUG to see details.",
                UserWarning
            )

        return samples, failures

    def _is_duplicate(self, text_hash: str) -> bool:
        """Check if a sample with this hash already exists."""
        return any(s.text_hash == text_hash for s in self.samples)

    def _update_aggregated_features(self) -> None:
        """
        Recalculate aggregated features from all samples.

        Computes mean, std, min, max for each feature across samples.
        """
        if not self.samples:
            self.aggregated_features = {}
            self.feature_statistics = {}
            return

        all_features: Dict[str, List[float]] = {}

        for sample in self.samples:
            if sample.analysis:
                flat = sample.analysis.get_flat_features()
                for key, value in flat.items():
                    if key not in all_features:
                        all_features[key] = []
                    all_features[key].append(value)

        self.aggregated_features = {}
        self.feature_statistics = {}

        for key, values in all_features.items():
            arr = np.array(values)
            self.aggregated_features[key] = float(np.mean(arr))
            self.feature_statistics[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(values),
            }

    def get_feature_vector(self) -> np.ndarray:
        """
        Get the aggregated feature vector for this profile.

        Returns features as a numpy array in sorted key order.
        """
        if not self.aggregated_features:
            return np.array([])

        sorted_keys = sorted(self.aggregated_features.keys())
        return np.array([self.aggregated_features[k] for k in sorted_keys])

    def get_feature_names(self) -> List[str]:
        """Get sorted list of feature names."""
        return sorted(self.aggregated_features.keys())

    def get_sample_count(self) -> int:
        """Get number of samples in profile."""
        return len(self.samples)

    def get_total_word_count(self) -> int:
        """Get total word count across all samples."""
        return sum(
            s.analysis.word_count
            for s in self.samples
            if s.analysis
        )

    def remove_sample(self, text_hash: str) -> bool:
        """
        Remove a sample by its hash.

        Returns True if sample was found and removed.
        """
        for i, sample in enumerate(self.samples):
            if sample.text_hash == text_hash:
                del self.samples[i]
                self.updated_at = datetime.now()
                self._update_aggregated_features()
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "name": self.name,
            "profile_id": self.profile_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "samples": [s.to_dict() for s in self.samples],
            "aggregated_features": self.aggregated_features,
            "feature_statistics": self.feature_statistics,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert profile to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthorProfile":
        """Create profile from dictionary."""
        profile = cls(
            name=data["name"],
            profile_id=data["profile_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            samples=[Sample.from_dict(s) for s in data.get("samples", [])],
            aggregated_features=data.get("aggregated_features", {}),
            feature_statistics=data.get("feature_statistics", {}),
            metadata=data.get("metadata", {}),
        )
        return profile

    @classmethod
    def from_json(cls, json_str: str) -> "AuthorProfile":
        """Create profile from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save(self, path: str, base_dir: Optional[str] = None) -> None:
        """
        Save profile to a JSON file.

        Args:
            path: File path to save to
            base_dir: Optional base directory for path traversal protection

        Raises:
            ValueError: If path traversal is detected
        """
        path = Path(path).resolve()
        if base_dir:
            base_path = Path(base_dir).resolve()
            if not str(path).startswith(str(base_path)):
                raise ValueError(f"Path traversal detected: {path} is outside {base_path}")

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str, base_dir: Optional[str] = None) -> "AuthorProfile":
        """
        Load profile from a JSON file.

        Args:
            path: File path to load from
            base_dir: Optional base directory for path traversal protection

        Returns:
            Loaded AuthorProfile

        Raises:
            ValueError: If path traversal is detected
        """
        resolved_path = Path(path).resolve()
        if base_dir:
            base_path = Path(base_dir).resolve()
            if not str(resolved_path).startswith(str(base_path)):
                raise ValueError(f"Path traversal detected: {resolved_path} is outside {base_path}")

        with open(resolved_path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())

    @classmethod
    def from_file(cls, path: str) -> "AuthorProfile":
        """
        Load profile from a file (alias for load).

        Args:
            path: File path to load from

        Returns:
            Loaded AuthorProfile
        """
        return cls.load(path)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the profile."""
        return {
            "name": self.name,
            "profile_id": self.profile_id,
            "sample_count": self.get_sample_count(),
            "total_words": self.get_total_word_count(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "feature_count": len(self.aggregated_features),
        }

    def get_distinctive_features(self, threshold_std: float = 1.0) -> List[Dict[str, Any]]:
        """
        Get features with low variance (distinctive/consistent patterns).

        Args:
            threshold_std: Maximum standard deviation to consider distinctive

        Returns:
            List of distinctive features with their statistics
        """
        distinctive = []

        for key, stats in self.feature_statistics.items():
            if stats["std"] <= threshold_std and stats["mean"] != 0:
                distinctive.append({
                    "feature": key,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "consistency": 1 - (stats["std"] / (abs(stats["mean"]) + 0.001)),
                })

        distinctive.sort(key=lambda x: x["consistency"], reverse=True)
        return distinctive


class ProfileManager:
    """Manages multiple author profiles."""

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize profile manager.

        Args:
            storage_dir: Directory for profile storage (optional)
        """
        self.profiles: Dict[str, AuthorProfile] = {}
        self.storage_dir = Path(storage_dir) if storage_dir else None

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_profiles()

    def _load_profiles(self) -> None:
        """Load all profiles from storage directory."""
        if not self.storage_dir:
            return

        load_errors = []
        for path in self.storage_dir.glob("*.json"):
            try:
                profile = AuthorProfile.load(str(path))
                self.profiles[profile.name] = profile
            except Exception as e:
                error_info = {"path": str(path), "error": f"{type(e).__name__}: {e}"}
                load_errors.append(error_info)
                logger.warning(f"Failed to load profile from {path}: {e}")

        if load_errors:
            warnings.warn(
                f"Failed to load {len(load_errors)} profiles from {self.storage_dir}. "
                f"Check logs for details.",
                UserWarning
            )

    def create_profile(
        self,
        name: str,
        samples: Optional[List[str]] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuthorProfile:
        """
        Create a new author profile.

        Args:
            name: Profile name
            samples: Optional list of initial text samples
            source: Source identifier for samples
            metadata: Profile metadata

        Returns:
            Created AuthorProfile
        """
        if name in self.profiles:
            raise ValueError(f"Profile '{name}' already exists")

        profile = AuthorProfile.create(name=name, metadata=metadata)

        if samples:
            analyzer = Analyzer()
            profile.add_samples(samples, source=source, analyzer=analyzer)

        self.profiles[name] = profile

        if self.storage_dir:
            self._save_profile(profile)

        return profile

    def _save_profile(self, profile: AuthorProfile) -> None:
        """Save a profile to storage."""
        if self.storage_dir:
            path = self.storage_dir / f"{profile.profile_id}.json"
            profile.save(str(path))

    def get_profile(self, name: str) -> Optional[AuthorProfile]:
        """Get a profile by name."""
        return self.profiles.get(name)

    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all profiles with summaries."""
        return [p.get_summary() for p in self.profiles.values()]

    def delete_profile(self, name: str) -> bool:
        """Delete a profile."""
        if name not in self.profiles:
            return False

        profile = self.profiles[name]

        if self.storage_dir:
            path = self.storage_dir / f"{profile.profile_id}.json"
            if path.exists():
                path.unlink()

        del self.profiles[name]
        return True

    def merge_profiles(
        self,
        names: List[str],
        new_name: str,
    ) -> AuthorProfile:
        """
        Merge multiple profiles into a new profile.

        Args:
            names: Names of profiles to merge
            new_name: Name for the merged profile

        Returns:
            New merged AuthorProfile
        """
        if new_name in self.profiles:
            raise ValueError(f"Profile '{new_name}' already exists")

        merged = AuthorProfile.create(name=new_name)

        for name in names:
            profile = self.profiles.get(name)
            if profile:
                merged.samples.extend(profile.samples)

        merged._update_aggregated_features()
        merged.updated_at = datetime.now()

        self.profiles[new_name] = merged

        if self.storage_dir:
            self._save_profile(merged)

        return merged
