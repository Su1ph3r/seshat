"""
Integrity verification for forensic evidence.

Provides hash verification and tamper detection.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import json


@dataclass
class VerificationResult:
    """Result of integrity verification."""
    verified: bool
    expected_hash: str
    computed_hash: str
    verified_at: datetime
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "verified": self.verified,
            "expected_hash": self.expected_hash,
            "computed_hash": self.computed_hash,
            "verified_at": self.verified_at.isoformat(),
            "details": self.details,
        }


class IntegrityVerifier:
    """
    Verify integrity of evidence and data.
    """

    def __init__(self, algorithm: str = "sha256"):
        """
        Initialize verifier.

        Args:
            algorithm: Hash algorithm to use
        """
        self.algorithm = algorithm

    def compute_hash(self, content: str) -> str:
        """
        Compute hash of content.

        Args:
            content: Content to hash

        Returns:
            Hex digest of hash
        """
        h = hashlib.new(self.algorithm)
        h.update(content.encode("utf-8"))
        return h.hexdigest()

    def compute_file_hash(self, file_path: str) -> str:
        """
        Compute hash of file.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of hash
        """
        h = hashlib.new(self.algorithm)

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)

        return h.hexdigest()

    def verify_content(
        self,
        content: str,
        expected_hash: str,
    ) -> VerificationResult:
        """
        Verify content against expected hash.

        Args:
            content: Content to verify
            expected_hash: Expected hash value

        Returns:
            VerificationResult
        """
        computed = self.compute_hash(content)

        return VerificationResult(
            verified=computed == expected_hash,
            expected_hash=expected_hash,
            computed_hash=computed,
            verified_at=datetime.now(),
            details={
                "algorithm": self.algorithm,
                "content_length": len(content),
            },
        )

    def verify_file(
        self,
        file_path: str,
        expected_hash: str,
    ) -> VerificationResult:
        """
        Verify file against expected hash.

        Args:
            file_path: Path to file
            expected_hash: Expected hash value

        Returns:
            VerificationResult
        """
        path = Path(file_path)
        computed = self.compute_file_hash(file_path)

        return VerificationResult(
            verified=computed == expected_hash,
            expected_hash=expected_hash,
            computed_hash=computed,
            verified_at=datetime.now(),
            details={
                "algorithm": self.algorithm,
                "file_path": str(path),
                "file_size": path.stat().st_size if path.exists() else 0,
            },
        )

    def verify_evidence_chain(
        self,
        chain_file: str,
    ) -> Dict[str, VerificationResult]:
        """
        Verify all items in an evidence chain.

        Args:
            chain_file: Path to evidence chain JSON

        Returns:
            Dictionary of evidence_id to VerificationResult
        """
        data = json.loads(Path(chain_file).read_text())
        results = {}

        for evidence_id, item in data.get("items", {}).items():
            content = item.get("content", "")
            expected = item.get("content_hash", "")

            results[evidence_id] = self.verify_content(content, expected)

        return results

    def generate_hash_manifest(
        self,
        directory: str,
        recursive: bool = True,
    ) -> Dict[str, str]:
        """
        Generate hash manifest for directory.

        Args:
            directory: Directory to hash
            recursive: Include subdirectories

        Returns:
            Dictionary mapping file paths to hashes
        """
        path = Path(directory)
        manifest = {}

        pattern = "**/*" if recursive else "*"

        for file_path in path.glob(pattern):
            if file_path.is_file():
                rel_path = file_path.relative_to(path)
                manifest[str(rel_path)] = self.compute_file_hash(str(file_path))

        return manifest

    def verify_manifest(
        self,
        directory: str,
        manifest: Dict[str, str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Verify directory against manifest.

        Args:
            directory: Directory to verify
            manifest: Expected hashes

        Returns:
            Tuple of (verified, modified, missing) file lists
        """
        path = Path(directory)
        verified = []
        modified = []
        missing = []

        for rel_path, expected_hash in manifest.items():
            file_path = path / rel_path

            if not file_path.exists():
                missing.append(rel_path)
                continue

            computed = self.compute_file_hash(str(file_path))

            if computed == expected_hash:
                verified.append(rel_path)
            else:
                modified.append(rel_path)

        return verified, modified, missing

    def create_signed_hash(
        self,
        content: str,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Create a timestamped signed hash.

        Args:
            content: Content to hash
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Signed hash dictionary
        """
        ts = timestamp or datetime.now()
        content_hash = self.compute_hash(content)

        combined = f"{content_hash}|{ts.isoformat()}"
        signature_hash = self.compute_hash(combined)

        return {
            "content_hash": content_hash,
            "timestamp": ts.isoformat(),
            "signature": signature_hash,
            "algorithm": self.algorithm,
        }

    def verify_signed_hash(
        self,
        content: str,
        signed_hash: Dict[str, Any],
    ) -> bool:
        """
        Verify a signed hash.

        Args:
            content: Content to verify
            signed_hash: Signed hash to verify against

        Returns:
            True if verified
        """
        content_hash = self.compute_hash(content)

        if content_hash != signed_hash.get("content_hash"):
            return False

        ts = signed_hash.get("timestamp")
        combined = f"{content_hash}|{ts}"
        expected_signature = self.compute_hash(combined)

        return expected_signature == signed_hash.get("signature")
