"""
Token-bucket rate limiter — per IP, per model endpoint.

Uses a sliding token-bucket algorithm:
  - Each IP starts with `burst` tokens (= 2 × rps by default).
  - Tokens refill at `rps` per second continuously.
  - Each request consumes 1 token.
  - If the bucket is empty → return False (caller sends 429).

Thread-safe (protects each bucket with a per-IP lock).
Buckets are pruned after `ttl_s` seconds of inactivity to prevent unbounded growth.

Usage
─────
    limiter = RateLimiter(rps=10)          # 10 req/s per IP
    allowed, retry_after = limiter.check("1.2.3.4")
    if not allowed:
        return HTTP_429(Retry-After=retry_after)
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Tuple


class _Bucket:
    __slots__ = ("tokens", "last_refill", "lock")

    def __init__(self, capacity: float) -> None:
        self.tokens: float = capacity
        self.last_refill: float = time.monotonic()
        self.lock = threading.Lock()


class RateLimiter:
    """Per-IP token-bucket rate limiter."""

    def __init__(
        self,
        rps: float,
        burst: float = 0.0,
        ttl_s: float = 120.0,
    ) -> None:
        """
        Parameters
        ----------
        rps:
            Sustained requests per second allowed per IP.
        burst:
            Bucket capacity (peak burst). Defaults to 2 × rps.
        ttl_s:
            Seconds of inactivity before an IP's bucket is evicted.
        """
        self._rps = rps
        self._capacity = burst if burst > 0 else max(2.0, rps * 2)
        self._ttl = ttl_s
        self._buckets: Dict[str, _Bucket] = {}
        self._global_lock = threading.Lock()
        self._last_prune = time.monotonic()

    def check(self, ip: str) -> Tuple[bool, float]:
        """
        Consume one token for `ip`.

        Returns
        -------
        (allowed, retry_after_s)
            allowed=True if the request is permitted.
            retry_after_s is the number of seconds until the next token
            is available (only meaningful when allowed=False).
        """
        now = time.monotonic()
        self._maybe_prune(now)

        bucket = self._get_or_create(ip, now)
        with bucket.lock:
            elapsed = now - bucket.last_refill
            bucket.tokens = min(
                self._capacity,
                bucket.tokens + elapsed * self._rps,
            )
            bucket.last_refill = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0
            else:
                # Time until next token is available.
                retry_after = (1.0 - bucket.tokens) / self._rps
                return False, retry_after

    def _get_or_create(self, ip: str, now: float) -> _Bucket:
        with self._global_lock:
            if ip not in self._buckets:
                self._buckets[ip] = _Bucket(self._capacity)
            return self._buckets[ip]

    def _maybe_prune(self, now: float) -> None:
        """Evict inactive buckets every ttl_s seconds."""
        if now - self._last_prune < self._ttl:
            return
        with self._global_lock:
            cutoff = now - self._ttl
            dead = [
                ip for ip, b in self._buckets.items()
                if b.last_refill < cutoff
            ]
            for ip in dead:
                del self._buckets[ip]
            self._last_prune = now
