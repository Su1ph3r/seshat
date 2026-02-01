"""
Base scraper class for Seshat.

Provides abstract interface for platform-specific scrapers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Generator
from datetime import datetime
import hashlib


@dataclass
class ScrapedContent:
    """
    Represents scraped content from any platform.
    """
    text: str
    source_url: str
    platform: str
    author: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = field(default="", init=False)

    def __post_init__(self):
        """Compute content hash after initialization."""
        self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of content."""
        content = f"{self.text}|{self.source_url}|{self.platform}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "source_url": self.source_url,
            "platform": self.platform,
            "author": self.author,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScrapedContent":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            text=data["text"],
            source_url=data["source_url"],
            platform=data["platform"],
            author=data.get("author"),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


class BaseScraper(ABC):
    """
    Abstract base class for all scrapers.

    Provides common functionality and interface for platform-specific scrapers.
    """

    def __init__(
        self,
        rate_limit: float = 1.0,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """
        Initialize base scraper.

        Args:
            rate_limit: Minimum seconds between requests
            proxy: Optional proxy URL
            user_agent: Optional custom user agent
        """
        self.rate_limit = rate_limit
        self.proxy = proxy
        self.user_agent = user_agent or self._default_user_agent()
        self._last_request_time = 0

    def _default_user_agent(self) -> str:
        """Return default user agent."""
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return platform name."""
        pass

    @abstractmethod
    def scrape_user(
        self,
        username: str,
        limit: int = 100,
        **kwargs,
    ) -> Generator[ScrapedContent, None, None]:
        """
        Scrape content from a user's account.

        Args:
            username: Username to scrape
            limit: Maximum number of items to scrape
            **kwargs: Additional platform-specific arguments

        Yields:
            ScrapedContent objects
        """
        pass

    @abstractmethod
    def scrape_search(
        self,
        query: str,
        limit: int = 100,
        **kwargs,
    ) -> Generator[ScrapedContent, None, None]:
        """
        Scrape content matching a search query.

        Args:
            query: Search query
            limit: Maximum number of items to scrape
            **kwargs: Additional platform-specific arguments

        Yields:
            ScrapedContent objects
        """
        pass

    def scrape_url(self, url: str) -> Optional[ScrapedContent]:
        """
        Scrape content from a specific URL.

        Args:
            url: URL to scrape

        Returns:
            ScrapedContent or None if failed
        """
        raise NotImplementedError("URL scraping not implemented for this platform")

    def scrape_user_to_list(
        self,
        username: str,
        limit: int = 100,
        **kwargs,
    ) -> List[ScrapedContent]:
        """
        Scrape user content and return as list.

        Args:
            username: Username to scrape
            limit: Maximum number of items
            **kwargs: Additional arguments

        Returns:
            List of ScrapedContent
        """
        return list(self.scrape_user(username, limit, **kwargs))

    def scrape_search_to_list(
        self,
        query: str,
        limit: int = 100,
        **kwargs,
    ) -> List[ScrapedContent]:
        """
        Scrape search results and return as list.

        Args:
            query: Search query
            limit: Maximum number of items
            **kwargs: Additional arguments

        Returns:
            List of ScrapedContent
        """
        return list(self.scrape_search(query, limit, **kwargs))

    def _apply_rate_limit(self):
        """Apply rate limiting between requests."""
        import time

        current_time = time.time()
        elapsed = current_time - self._last_request_time

        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        self._last_request_time = time.time()

    def _get_session(self):
        """Get HTTP session with configured proxy and headers."""
        import httpx

        headers = {"User-Agent": self.user_agent}
        proxies = {"http://": self.proxy, "https://": self.proxy} if self.proxy else None

        return httpx.Client(headers=headers, proxies=proxies, timeout=30.0)
