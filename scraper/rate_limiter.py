"""
Rate limiter for web scraping.

Provides token bucket and sliding window rate limiting.
"""

from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import threading


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 1.0
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None
    burst_size: int = 1


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.

    Allows bursts of requests up to bucket capacity,
    then enforces steady rate.
    """

    def __init__(
        self,
        rate: float = 1.0,
        capacity: int = 1,
    ):
        """
        Initialize token bucket.

        Args:
            rate: Tokens per second
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time waited in seconds
        """
        with self._lock:
            wait_time = 0

            self._refill()

            if self._tokens < tokens:
                deficit = tokens - self._tokens
                wait_time = deficit / self.rate
                time.sleep(wait_time)
                self._refill()

            self._tokens -= tokens
            return wait_time

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(
            self.capacity,
            self._tokens + elapsed * self.rate,
        )
        self._last_update = now


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    Tracks requests in time windows for more precise limiting.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
    ):
        """
        Initialize sliding window limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: list = []
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """
        Acquire permission to make a request.

        Returns:
            Time waited in seconds
        """
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds

            self._requests = [t for t in self._requests if t > cutoff]

            if len(self._requests) >= self.max_requests:
                oldest = self._requests[0]
                wait_time = oldest + self.window_seconds - now
                if wait_time > 0:
                    time.sleep(wait_time)
                    now = time.time()
                    cutoff = now - self.window_seconds
                    self._requests = [t for t in self._requests if t > cutoff]

            self._requests.append(now)
            return 0 if len(self._requests) < self.max_requests else wait_time

    def remaining(self) -> int:
        """Return remaining requests in current window."""
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            current = len([t for t in self._requests if t > cutoff])
            return max(0, self.max_requests - current)


class RateLimiter:
    """
    Combined rate limiter with multiple strategies.

    Supports per-second, per-minute, and per-hour limits.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._limiters: Dict[str, SlidingWindowRateLimiter] = {}

        self._token_bucket = TokenBucketRateLimiter(
            rate=self.config.requests_per_second,
            capacity=self.config.burst_size,
        )

        if self.config.requests_per_minute:
            self._limiters["minute"] = SlidingWindowRateLimiter(
                max_requests=self.config.requests_per_minute,
                window_seconds=60,
            )

        if self.config.requests_per_hour:
            self._limiters["hour"] = SlidingWindowRateLimiter(
                max_requests=self.config.requests_per_hour,
                window_seconds=3600,
            )

    def acquire(self) -> float:
        """
        Acquire permission to make a request.

        Returns:
            Total time waited in seconds
        """
        total_wait = 0

        for limiter in self._limiters.values():
            total_wait += limiter.acquire()

        total_wait += self._token_bucket.acquire()

        return total_wait

    def wait(self):
        """Wait for rate limit (alias for acquire)."""
        self.acquire()

    @classmethod
    def for_twitter(cls) -> "RateLimiter":
        """Create rate limiter with Twitter-appropriate settings."""
        return cls(RateLimitConfig(
            requests_per_second=0.5,
            requests_per_minute=15,
            requests_per_hour=450,
            burst_size=3,
        ))

    @classmethod
    def for_reddit(cls) -> "RateLimiter":
        """Create rate limiter with Reddit-appropriate settings."""
        return cls(RateLimitConfig(
            requests_per_second=1.0,
            requests_per_minute=30,
            burst_size=5,
        ))

    @classmethod
    def for_generic(cls) -> "RateLimiter":
        """Create conservative rate limiter for generic websites."""
        return cls(RateLimitConfig(
            requests_per_second=0.5,
            requests_per_minute=20,
            burst_size=2,
        ))
