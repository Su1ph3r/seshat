"""
Evidence chain management for forensic analysis.

Maintains chain of custody for collected evidence.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid


@dataclass
class CustodyRecord:
    """Record of custody transfer."""
    timestamp: datetime
    action: str
    custodian: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "custodian": self.custodian,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustodyRecord":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action=data["action"],
            custodian=data["custodian"],
            notes=data.get("notes"),
        )


@dataclass
class EvidenceItem:
    """
    Represents a piece of evidence with chain of custody.
    """
    evidence_id: str
    content: str
    content_hash: str
    source_url: str
    source_platform: str
    collected_at: datetime
    collected_by: str
    collection_method: str
    original_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    chain_of_custody: List[CustodyRecord] = field(default_factory=list)

    def __post_init__(self):
        """Add initial custody record."""
        if not self.chain_of_custody:
            self.chain_of_custody.append(CustodyRecord(
                timestamp=self.collected_at,
                action="collected",
                custodian=self.collected_by,
                notes=f"Collected via {self.collection_method}",
            ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "content": self.content,
            "content_hash": self.content_hash,
            "source_url": self.source_url,
            "source_platform": self.source_platform,
            "collected_at": self.collected_at.isoformat(),
            "collected_by": self.collected_by,
            "collection_method": self.collection_method,
            "original_timestamp": self.original_timestamp.isoformat() if self.original_timestamp else None,
            "metadata": self.metadata,
            "chain_of_custody": [r.to_dict() for r in self.chain_of_custody],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceItem":
        """Create from dictionary."""
        chain = [CustodyRecord.from_dict(r) for r in data.get("chain_of_custody", [])]

        original_ts = data.get("original_timestamp")
        if original_ts and isinstance(original_ts, str):
            original_ts = datetime.fromisoformat(original_ts)

        item = cls(
            evidence_id=data["evidence_id"],
            content=data["content"],
            content_hash=data["content_hash"],
            source_url=data["source_url"],
            source_platform=data["source_platform"],
            collected_at=datetime.fromisoformat(data["collected_at"]),
            collected_by=data["collected_by"],
            collection_method=data["collection_method"],
            original_timestamp=original_ts,
            metadata=data.get("metadata", {}),
            chain_of_custody=chain,
        )
        return item

    def add_custody_record(
        self,
        action: str,
        custodian: str,
        notes: Optional[str] = None,
    ):
        """Add a custody record."""
        self.chain_of_custody.append(CustodyRecord(
            timestamp=datetime.now(),
            action=action,
            custodian=custodian,
            notes=notes,
        ))

    def verify_integrity(self) -> bool:
        """Verify content hash matches."""
        computed = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        return computed == self.content_hash


class EvidenceChain:
    """
    Manages a collection of evidence items with chain of custody.
    """

    def __init__(
        self,
        case_id: Optional[str] = None,
        case_name: Optional[str] = None,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize evidence chain.

        Args:
            case_id: Unique case identifier
            case_name: Human-readable case name
            storage_path: Path to store evidence
        """
        self.case_id = case_id or str(uuid.uuid4())
        self.case_name = case_name or f"Case-{self.case_id[:8]}"
        self.created_at = datetime.now()
        self.storage_path = Path(storage_path) if storage_path else None
        self._items: Dict[str, EvidenceItem] = {}

    def collect(
        self,
        content: str,
        source_url: str,
        source_platform: str,
        collected_by: str,
        collection_method: str,
        original_timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceItem:
        """
        Collect and register evidence.

        Args:
            content: Evidence content
            source_url: Source URL
            source_platform: Platform name
            collected_by: Collector identity
            collection_method: How content was collected
            original_timestamp: Original creation time
            metadata: Additional metadata

        Returns:
            Created EvidenceItem
        """
        evidence_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        item = EvidenceItem(
            evidence_id=evidence_id,
            content=content,
            content_hash=content_hash,
            source_url=source_url,
            source_platform=source_platform,
            collected_at=datetime.now(),
            collected_by=collected_by,
            collection_method=collection_method,
            original_timestamp=original_timestamp,
            metadata=metadata or {},
        )

        self._items[evidence_id] = item

        if self.storage_path:
            self._save_item(item)

        return item

    def get(self, evidence_id: str) -> Optional[EvidenceItem]:
        """Get evidence item by ID."""
        return self._items.get(evidence_id)

    def list_all(self) -> List[EvidenceItem]:
        """List all evidence items."""
        return list(self._items.values())

    def verify_all(self) -> Dict[str, bool]:
        """
        Verify integrity of all evidence.

        Returns:
            Dictionary mapping evidence_id to verification result
        """
        return {
            eid: item.verify_integrity()
            for eid, item in self._items.items()
        }

    def export(self, output_path: str, base_dir: Optional[str] = None):
        """
        Export evidence chain to JSON file.

        Args:
            output_path: Path to save export
            base_dir: Optional base directory for path traversal protection

        Raises:
            ValueError: If path traversal is detected
        """
        resolved_path = Path(output_path).resolve()
        if base_dir:
            base_path = Path(base_dir).resolve()
            if not str(resolved_path).startswith(str(base_path)):
                raise ValueError(f"Path traversal detected: {resolved_path} is outside {base_path}")

        data = {
            "case_id": self.case_id,
            "case_name": self.case_name,
            "created_at": self.created_at.isoformat(),
            "exported_at": datetime.now().isoformat(),
            "evidence_count": len(self._items),
            "items": {eid: item.to_dict() for eid, item in self._items.items()},
        }

        chain_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        data["chain_hash"] = chain_hash

        resolved_path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, input_path: str, base_dir: Optional[str] = None) -> "EvidenceChain":
        """
        Load evidence chain from JSON file.

        Args:
            input_path: Path to load from
            base_dir: Optional base directory for path traversal protection

        Returns:
            Loaded EvidenceChain

        Raises:
            ValueError: If path traversal is detected
        """
        resolved_path = Path(input_path).resolve()
        if base_dir:
            base_path = Path(base_dir).resolve()
            if not str(resolved_path).startswith(str(base_path)):
                raise ValueError(f"Path traversal detected: {resolved_path} is outside {base_path}")

        data = json.loads(resolved_path.read_text())

        chain = cls(
            case_id=data["case_id"],
            case_name=data["case_name"],
        )
        chain.created_at = datetime.fromisoformat(data["created_at"])

        for eid, item_data in data.get("items", {}).items():
            chain._items[eid] = EvidenceItem.from_dict(item_data)

        return chain

    def _save_item(self, item: EvidenceItem):
        """Save individual item to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        item_path = self.storage_path / f"{item.evidence_id}.json"
        item_path.write_text(json.dumps(item.to_dict(), indent=2))

    def generate_manifest(self) -> Dict[str, Any]:
        """
        Generate evidence manifest.

        Returns:
            Manifest dictionary
        """
        return {
            "case_id": self.case_id,
            "case_name": self.case_name,
            "created_at": self.created_at.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "evidence_count": len(self._items),
            "platforms": list(set(i.source_platform for i in self._items.values())),
            "items": [
                {
                    "evidence_id": item.evidence_id,
                    "content_hash": item.content_hash,
                    "source_url": item.source_url,
                    "collected_at": item.collected_at.isoformat(),
                    "custody_count": len(item.chain_of_custody),
                }
                for item in self._items.values()
            ],
        }
