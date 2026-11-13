"""Rate limits and daily caps with shared state.

``UpstashLimiter`` keeps counters in Upstash Redis over its REST API, so every serverless instance sees the
same numbers. ``MemoryLimiter`` keeps them in the process: correct on one long-lived server, but on a
serverless platform each instance has its own counters, so the limits are per instance and reset when an
instance is recycled. ``make_limiter`` picks Upstash when its environment variables are present and falls back
to memory when they are not or when Upstash cannot be reached.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Protocol

import httpx

from .config import AskSettings


class Limiter(Protocol):
    mode: str

    def hit(self, key: str, ttl: int, n: int = 1) -> int: ...

    def take_global(self, cap: int) -> bool: ...

    def take_provider(self, name: str, cap: int) -> bool: ...

    def tokens_window(self, name: str, now: float) -> list[tuple[int, int]]: ...

    def tokens_today(self, name: str, now: float) -> int: ...

    def add_tokens(self, name: str, n: int, now: float) -> int: ...

    def correct_tokens(self, name: str, delta: int, bucket_start: int, now: float) -> None: ...

    def check_ip(self, ip: str, per_minute: int, per_day: int) -> tuple[bool, int]: ...


WINDOW_BUCKET = 10
WINDOW_BUCKETS = 6


def _day(now: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(now))


class _Base:
    mode = "memory"

    def __init__(self, clock=time.time, salt: str = "ask-aditya") -> None:
        self.clock = clock
        self.salt = salt

    def hit(self, key: str, ttl: int, n: int = 1) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def check_ip(self, ip: str, per_minute: int, per_day: int) -> tuple[bool, int]:
        """Count one request from this address. Returns (allowed, retry_after_seconds)."""
        now = self.clock()
        h = hashlib.sha256(f"{self.salt}:{ip}".encode()).hexdigest()[:16]
        minute = self.hit(f"ip:{h}:m:{int(now // 60)}", 90)
        day = self.hit(f"ip:{h}:d:{_day(now)}", 90000)
        if minute > per_minute:
            return False, int(60 - now % 60) + 1
        if day > per_day:
            return False, int(86400 - now % 86400) + 1
        return True, 0

    def take_global(self, cap: int) -> bool:
        return self.hit(f"g:{_day(self.clock())}", 90000) <= cap

    def take_provider(self, name: str, cap: int) -> bool:
        return self.hit(f"p:{name}:{_day(self.clock())}", 90000) <= cap

    # ---- token accounting: a sliding 60 s window made of 10 s buckets, and a daily total -------------------
    def peek_many(self, keys: list[str]) -> list[int]:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def window_starts(now: float) -> list[int]:
        """Start times of the six 10-second buckets that make up the last minute, oldest first."""
        cur = int(now // WINDOW_BUCKET) * WINDOW_BUCKET
        return [cur - WINDOW_BUCKET * i for i in range(WINDOW_BUCKETS - 1, -1, -1)]

    def tokens_window(self, name: str, now: float) -> list[tuple[int, int]]:
        """(bucket start, tokens) for the last minute: real usage plus the reservations of calls in flight."""
        starts = self.window_starts(now)
        counts = self.peek_many([f"t:{name}:w:{s}" for s in starts])
        return list(zip(starts, counts, strict=True))

    def tokens_today(self, name: str, now: float) -> int:
        return self.peek_many([f"t:{name}:d:{_day(now)}"])[0]

    def add_tokens(self, name: str, n: int, now: float) -> int:
        """Add ``n`` tokens (negative to correct) to the bucket of ``now`` and to the day; returns the bucket start."""
        start = int(now // WINDOW_BUCKET) * WINDOW_BUCKET
        self.hit(f"t:{name}:w:{start}", WINDOW_BUCKET * (WINDOW_BUCKETS + 1), n)
        self.hit(f"t:{name}:d:{_day(now)}", 90000, n)
        return start

    def correct_tokens(self, name: str, delta: int, bucket_start: int, now: float) -> None:
        """Replace a reservation by the real usage in the bucket it was made in."""
        if delta:
            self.hit(f"t:{name}:w:{bucket_start}", WINDOW_BUCKET * (WINDOW_BUCKETS + 1), delta)
            self.hit(f"t:{name}:d:{_day(now)}", 90000, delta)


class MemoryLimiter(_Base):
    mode = "memory"

    def __init__(self, clock=time.time, salt: str = "ask-aditya") -> None:
        super().__init__(clock, salt)
        self._d: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, ttl: int, n: int = 1) -> int:
        now = self.clock()
        with self._lock:
            if len(self._d) > 5000:
                self._d = {k: v for k, v in self._d.items() if v[1] > now}
            count, exp = self._d.get(key, (0, 0.0))
            if exp <= now:
                count, exp = 0, now + ttl
            count += n
            self._d[key] = (count, exp)
            return count

    def peek_many(self, keys: list[str]) -> list[int]:
        now = self.clock()
        with self._lock:
            return [self._d[k][0] if k in self._d and self._d[k][1] > now else 0 for k in keys]


class UpstashLimiter(_Base):
    mode = "upstash"

    def __init__(
        self,
        url: str,
        token: str,
        fallback: MemoryLimiter,
        clock=time.time,
        salt: str = "ask-aditya",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(clock, salt)
        self.url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._client = httpx.Client(transport=transport, timeout=2.0)
        self._fallback = fallback

    def hit(self, key: str, ttl: int, n: int = 1) -> int:
        k = f"ask:{key}"
        try:
            r = self._client.post(
                f"{self.url}/pipeline",
                json=[["INCRBY", k, n], ["EXPIRE", k, ttl, "NX"]],
                headers=self._headers,
            )
            r.raise_for_status()
            return int(r.json()[0]["result"])
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError):
            return self._fallback.hit(
                key, ttl, n
            )  # Upstash unreachable: per-instance counting rather than none

    def peek_many(self, keys: list[str]) -> list[int]:
        try:
            r = self._client.post(
                f"{self.url}/pipeline",
                json=[["GET", f"ask:{k}"] for k in keys],
                headers=self._headers,
            )
            r.raise_for_status()
            return [int(x["result"] or 0) for x in r.json()]
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError):
            return self._fallback.peek_many(keys)


def make_limiter(
    settings: AskSettings, clock=time.time, transport: httpx.BaseTransport | None = None
) -> Limiter:
    mem = MemoryLimiter(clock, settings.ip_salt)
    if settings.upstash_url and settings.upstash_token:
        return UpstashLimiter(
            settings.upstash_url, settings.upstash_token, mem, clock, settings.ip_salt, transport
        )
    return mem
