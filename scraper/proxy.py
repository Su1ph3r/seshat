"""
Proxy management for web scraping.

Provides proxy rotation and validation.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import random
import time
import threading


@dataclass
class Proxy:
    """Represents a proxy server."""
    url: str
    protocol: str = "http"
    username: Optional[str] = None
    password: Optional[str] = None
    country: Optional[str] = None
    last_used: float = 0
    failures: int = 0
    successes: int = 0

    @property
    def full_url(self) -> str:
        """Get full proxy URL with auth."""
        if self.username and self.password:
            # Parse URL parts
            if "://" in self.url:
                protocol, rest = self.url.split("://", 1)
                return f"{protocol}://{self.username}:{self.password}@{rest}"
            return f"{self.protocol}://{self.username}:{self.password}@{self.url}"
        return self.url if "://" in self.url else f"{self.protocol}://{self.url}"

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5

    def mark_success(self):
        """Mark successful request through this proxy."""
        self.successes += 1
        self.last_used = time.time()

    def mark_failure(self):
        """Mark failed request through this proxy."""
        self.failures += 1
        self.last_used = time.time()


class ProxyManager:
    """
    Manages a pool of proxy servers with rotation.
    """

    def __init__(
        self,
        proxies: Optional[List[str]] = None,
        rotation_strategy: str = "round_robin",
        min_success_rate: float = 0.3,
        cooldown_seconds: float = 5.0,
    ):
        """
        Initialize proxy manager.

        Args:
            proxies: List of proxy URLs
            rotation_strategy: "round_robin", "random", or "weighted"
            min_success_rate: Minimum success rate to keep using proxy
            cooldown_seconds: Minimum time between using same proxy
        """
        self._proxies: List[Proxy] = []
        self._current_index = 0
        self._lock = threading.Lock()
        self.rotation_strategy = rotation_strategy
        self.min_success_rate = min_success_rate
        self.cooldown_seconds = cooldown_seconds

        if proxies:
            for p in proxies:
                self.add_proxy(p)

    def add_proxy(
        self,
        url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        country: Optional[str] = None,
    ):
        """
        Add a proxy to the pool.

        Args:
            url: Proxy URL
            username: Optional auth username
            password: Optional auth password
            country: Optional country code
        """
        proxy = Proxy(
            url=url,
            username=username,
            password=password,
            country=country,
        )
        with self._lock:
            self._proxies.append(proxy)

    def add_proxies_from_file(self, filepath: str):
        """
        Load proxies from file (one per line).

        Format: protocol://host:port or protocol://user:pass@host:port
        """
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    self.add_proxy(line)

    def get_proxy(self) -> Optional[str]:
        """
        Get next proxy based on rotation strategy.

        Returns:
            Proxy URL or None if no proxies available
        """
        with self._lock:
            if not self._proxies:
                return None

            available = self._get_available_proxies()
            if not available:
                return None

            if self.rotation_strategy == "round_robin":
                proxy = self._round_robin(available)
            elif self.rotation_strategy == "random":
                proxy = self._random(available)
            elif self.rotation_strategy == "weighted":
                proxy = self._weighted(available)
            else:
                proxy = available[0]

            proxy.last_used = time.time()
            return proxy.full_url

    def _get_available_proxies(self) -> List[Proxy]:
        """Get proxies that are available for use."""
        now = time.time()
        return [
            p for p in self._proxies
            if p.success_rate >= self.min_success_rate
            and (now - p.last_used) >= self.cooldown_seconds
        ]

    def _round_robin(self, proxies: List[Proxy]) -> Proxy:
        """Round robin selection."""
        self._current_index = (self._current_index + 1) % len(proxies)
        return proxies[self._current_index]

    def _random(self, proxies: List[Proxy]) -> Proxy:
        """Random selection."""
        return random.choice(proxies)

    def _weighted(self, proxies: List[Proxy]) -> Proxy:
        """Weighted selection based on success rate."""
        weights = [p.success_rate + 0.1 for p in proxies]
        return random.choices(proxies, weights=weights, k=1)[0]

    def mark_success(self, proxy_url: str):
        """Mark a proxy as successful."""
        with self._lock:
            for p in self._proxies:
                if p.full_url == proxy_url:
                    p.mark_success()
                    break

    def mark_failure(self, proxy_url: str):
        """Mark a proxy as failed."""
        with self._lock:
            for p in self._proxies:
                if p.full_url == proxy_url:
                    p.mark_failure()
                    break

    def remove_proxy(self, proxy_url: str):
        """Remove a proxy from the pool."""
        with self._lock:
            self._proxies = [p for p in self._proxies if p.full_url != proxy_url]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about proxy pool."""
        with self._lock:
            total = len(self._proxies)
            healthy = len([p for p in self._proxies if p.success_rate >= self.min_success_rate])

            return {
                "total_proxies": total,
                "healthy_proxies": healthy,
                "proxies": [
                    {
                        "url": p.url,
                        "successes": p.successes,
                        "failures": p.failures,
                        "success_rate": p.success_rate,
                        "country": p.country,
                    }
                    for p in self._proxies
                ],
            }

    def __len__(self) -> int:
        """Return number of proxies in pool."""
        return len(self._proxies)

    def __bool__(self) -> bool:
        """Return True if proxies available."""
        return len(self._proxies) > 0
