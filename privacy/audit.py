"""
Audit logging for Seshat.

Tracks all operations for compliance and security.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import hashlib


@dataclass
class AuditEntry:
    """Represents an audit log entry."""
    timestamp: datetime
    action: str
    entity_type: str
    entity_id: Optional[str]
    user_id: Optional[str]
    details: Dict[str, Any]
    checksum: str = field(default="", init=False)

    def __post_init__(self):
        """Compute checksum after initialization."""
        self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute checksum for integrity verification."""
        content = f"{self.timestamp.isoformat()}|{self.action}|{self.entity_type}|{self.entity_id}|{self.user_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "user_id": self.user_id,
            "details": self.details,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEntry":
        """Create from dictionary."""
        entry = cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action=data["action"],
            entity_type=data["entity_type"],
            entity_id=data.get("entity_id"),
            user_id=data.get("user_id"),
            details=data.get("details", {}),
        )
        return entry


class AuditLogger:
    """
    Audit logger for tracking all operations.
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        max_entries: int = 10000,
    ):
        """
        Initialize audit logger.

        Args:
            log_file: Path to audit log file
            max_entries: Maximum entries to keep in memory
        """
        self.log_file = Path(log_file) if log_file else None
        self.max_entries = max_entries
        self._entries: List[AuditEntry] = []

        if self.log_file and self.log_file.exists():
            self._load()

    def _load(self):
        """Load entries from log file."""
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        self._entries.append(AuditEntry.from_dict(data))
                    except Exception:
                        continue

            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]

        except Exception:
            pass

    def _save_entry(self, entry: AuditEntry):
        """Append entry to log file."""
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")

    def log(
        self,
        action: str,
        entity_type: str,
        entity_id: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Log an action.

        Args:
            action: Action performed
            entity_type: Type of entity affected
            entity_id: ID of entity
            user_id: User who performed action
            details: Additional details

        Returns:
            Created AuditEntry
        """
        entry = AuditEntry(
            timestamp=datetime.now(),
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            details=details or {},
        )

        self._entries.append(entry)

        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

        self._save_entry(entry)

        return entry

    def log_profile_create(self, profile_id: str, name: str, user_id: Optional[str] = None):
        """Log profile creation."""
        return self.log(
            action="profile_create",
            entity_type="profile",
            entity_id=profile_id,
            user_id=user_id,
            details={"name": name},
        )

    def log_profile_delete(self, profile_id: str, user_id: Optional[str] = None):
        """Log profile deletion."""
        return self.log(
            action="profile_delete",
            entity_type="profile",
            entity_id=profile_id,
            user_id=user_id,
        )

    def log_sample_add(
        self,
        profile_id: str,
        sample_hash: str,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """Log sample addition."""
        return self.log(
            action="sample_add",
            entity_type="sample",
            entity_id=sample_hash,
            user_id=user_id,
            details={"profile_id": profile_id, "source": source},
        )

    def log_analysis(
        self,
        text_hash: str,
        profile_id: Optional[str] = None,
        score: Optional[float] = None,
        user_id: Optional[str] = None,
    ):
        """Log analysis operation."""
        return self.log(
            action="analysis",
            entity_type="text",
            entity_id=text_hash,
            user_id=user_id,
            details={"profile_id": profile_id, "score": score},
        )

    def log_export(
        self,
        profile_id: str,
        format: str,
        user_id: Optional[str] = None,
    ):
        """Log profile export."""
        return self.log(
            action="export",
            entity_type="profile",
            entity_id=profile_id,
            user_id=user_id,
            details={"format": format},
        )

    def log_scrape(
        self,
        platform: str,
        username: str,
        sample_count: int,
        user_id: Optional[str] = None,
    ):
        """Log scraping operation."""
        return self.log(
            action="scrape",
            entity_type="account",
            entity_id=f"{platform}:{username}",
            user_id=user_id,
            details={"sample_count": sample_count},
        )

    def query(
        self,
        action: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """
        Query audit log.

        Args:
            action: Filter by action
            entity_type: Filter by entity type
            entity_id: Filter by entity ID
            user_id: Filter by user
            since: Only entries after this time
            until: Only entries before this time
            limit: Maximum entries to return

        Returns:
            List of matching AuditEntry objects
        """
        results = self._entries.copy()

        if action:
            results = [e for e in results if e.action == action]

        if entity_type:
            results = [e for e in results if e.entity_type == entity_type]

        if entity_id:
            results = [e for e in results if e.entity_id == entity_id]

        if user_id:
            results = [e for e in results if e.user_id == user_id]

        if since:
            results = [e for e in results if e.timestamp >= since]

        if until:
            results = [e for e in results if e.timestamp <= until]

        results.sort(key=lambda e: e.timestamp, reverse=True)

        return results[:limit]

    def get_activity_summary(
        self,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get activity summary.

        Args:
            since: Start of period

        Returns:
            Summary dictionary
        """
        entries = self._entries
        if since:
            entries = [e for e in entries if e.timestamp >= since]

        action_counts = {}
        entity_counts = {}
        user_counts = {}

        for entry in entries:
            action_counts[entry.action] = action_counts.get(entry.action, 0) + 1
            entity_counts[entry.entity_type] = entity_counts.get(entry.entity_type, 0) + 1
            if entry.user_id:
                user_counts[entry.user_id] = user_counts.get(entry.user_id, 0) + 1

        return {
            "total_entries": len(entries),
            "action_counts": action_counts,
            "entity_counts": entity_counts,
            "user_counts": user_counts,
            "first_entry": entries[0].timestamp.isoformat() if entries else None,
            "last_entry": entries[-1].timestamp.isoformat() if entries else None,
        }

    def verify_integrity(self) -> List[AuditEntry]:
        """
        Verify integrity of all entries.

        Returns:
            List of entries with invalid checksums
        """
        invalid = []

        for entry in self._entries:
            expected = entry._compute_checksum()
            if expected != entry.checksum:
                invalid.append(entry)

        return invalid

    def export(self, output_path: str):
        """
        Export audit log to JSON file.

        Args:
            output_path: Output file path
        """
        data = {
            "exported_at": datetime.now().isoformat(),
            "entry_count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
