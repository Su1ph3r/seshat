"""
Account watcher for Seshat.

Monitors accounts for new content and style changes.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path


@dataclass
class WatchedAccount:
    """Represents a watched account."""
    platform: str
    username: str
    profile_name: Optional[str] = None
    interval_hours: float = 24.0
    last_check: Optional[datetime] = None
    last_sample_count: int = 0
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform": self.platform,
            "username": self.username,
            "profile_name": self.profile_name,
            "interval_hours": self.interval_hours,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_sample_count": self.last_sample_count,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatchedAccount":
        """Create from dictionary."""
        last_check = data.get("last_check")
        if last_check and isinstance(last_check, str):
            last_check = datetime.fromisoformat(last_check)

        return cls(
            platform=data["platform"],
            username=data["username"],
            profile_name=data.get("profile_name"),
            interval_hours=data.get("interval_hours", 24.0),
            last_check=last_check,
            last_sample_count=data.get("last_sample_count", 0),
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {}),
        )

    @property
    def is_due(self) -> bool:
        """Check if account is due for a check."""
        if not self.last_check:
            return True
        next_check = self.last_check + timedelta(hours=self.interval_hours)
        return datetime.now() >= next_check


class AccountWatcher:
    """
    Monitor accounts for new content and changes.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize account watcher.

        Args:
            storage_path: Path to store watchlist
        """
        self.storage_path = Path(storage_path) if storage_path else Path("~/.seshat/watchlist.json").expanduser()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._accounts: Dict[str, WatchedAccount] = {}
        self._load()

    def _load(self):
        """Load watchlist from storage."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for key, account_data in data.items():
                    self._accounts[key] = WatchedAccount.from_dict(account_data)
            except Exception:
                pass

    def _save(self):
        """Save watchlist to storage."""
        data = {key: acc.to_dict() for key, acc in self._accounts.items()}
        self.storage_path.write_text(json.dumps(data, indent=2))

    def _account_key(self, platform: str, username: str) -> str:
        """Generate unique key for account."""
        return f"{platform}:{username.lower()}"

    def add(
        self,
        platform: str,
        username: str,
        profile_name: Optional[str] = None,
        interval_hours: float = 24.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WatchedAccount:
        """
        Add account to watchlist.

        Args:
            platform: Platform name (twitter, reddit, etc.)
            username: Account username
            profile_name: Associated Seshat profile name
            interval_hours: Check interval in hours
            metadata: Additional metadata

        Returns:
            Created WatchedAccount
        """
        key = self._account_key(platform, username)

        account = WatchedAccount(
            platform=platform,
            username=username,
            profile_name=profile_name,
            interval_hours=interval_hours,
            metadata=metadata or {},
        )

        self._accounts[key] = account
        self._save()

        return account

    def remove(self, platform: str, username: str) -> bool:
        """
        Remove account from watchlist.

        Args:
            platform: Platform name
            username: Account username

        Returns:
            True if removed
        """
        key = self._account_key(platform, username)

        if key in self._accounts:
            del self._accounts[key]
            self._save()
            return True

        return False

    def get(self, platform: str, username: str) -> Optional[WatchedAccount]:
        """
        Get a watched account.

        Args:
            platform: Platform name
            username: Account username

        Returns:
            WatchedAccount or None
        """
        key = self._account_key(platform, username)
        return self._accounts.get(key)

    def list_all(self) -> List[WatchedAccount]:
        """
        List all watched accounts.

        Returns:
            List of WatchedAccount objects
        """
        return list(self._accounts.values())

    def list_due(self) -> List[WatchedAccount]:
        """
        List accounts due for checking.

        Returns:
            List of due WatchedAccount objects
        """
        return [acc for acc in self._accounts.values() if acc.is_active and acc.is_due]

    def update_last_check(
        self,
        platform: str,
        username: str,
        sample_count: int = 0,
    ):
        """
        Update last check time for an account.

        Args:
            platform: Platform name
            username: Account username
            sample_count: Number of samples collected
        """
        key = self._account_key(platform, username)

        if key in self._accounts:
            self._accounts[key].last_check = datetime.now()
            self._accounts[key].last_sample_count = sample_count
            self._save()

    def set_active(self, platform: str, username: str, active: bool):
        """
        Set account active status.

        Args:
            platform: Platform name
            username: Account username
            active: Active status
        """
        key = self._account_key(platform, username)

        if key in self._accounts:
            self._accounts[key].is_active = active
            self._save()

    def check_account(
        self,
        account: WatchedAccount,
        analyzer=None,
        profile_manager=None,
    ) -> Dict[str, Any]:
        """
        Check an account for new content.

        Args:
            account: Account to check
            analyzer: Optional Analyzer instance
            profile_manager: Optional ProfileManager instance

        Returns:
            Check results
        """
        from scraper import TwitterScraper, RedditScraper

        results = {
            "account": account.to_dict(),
            "checked_at": datetime.now().isoformat(),
            "new_samples": 0,
            "samples": [],
            "errors": [],
        }

        try:
            if account.platform == "twitter":
                scraper = TwitterScraper()
                samples = list(scraper.scrape_user(account.username, limit=50))

            elif account.platform == "reddit":
                scraper = RedditScraper()
                samples = list(scraper.scrape_user(account.username, limit=50))

            else:
                results["errors"].append(f"Unsupported platform: {account.platform}")
                return results

            results["new_samples"] = len(samples)
            results["samples"] = [s.to_dict() for s in samples[:10]]

            if account.profile_name and profile_manager and analyzer:
                profile = profile_manager.get_profile(account.profile_name)
                if profile:
                    for sample in samples:
                        try:
                            profile.add_sample(
                                sample.text,
                                source=account.platform,
                                analyzer=analyzer,
                            )
                        except Exception:
                            pass

                    profile_manager.save_profile(profile)

            self.update_last_check(account.platform, account.username, len(samples))

        except Exception as e:
            results["errors"].append(str(e))

        return results

    def run_checks(
        self,
        analyzer=None,
        profile_manager=None,
    ) -> List[Dict[str, Any]]:
        """
        Run checks for all due accounts.

        Args:
            analyzer: Optional Analyzer instance
            profile_manager: Optional ProfileManager instance

        Returns:
            List of check results
        """
        results = []

        for account in self.list_due():
            result = self.check_account(account, analyzer, profile_manager)
            results.append(result)

        return results
