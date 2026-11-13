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

    def take_tokens(
        self, name: str, n: int, per_minute: int | None, per_day: int | None
    ) -> bool: ...

    def adjust_tokens(self, name: str, delta: int) -> None: ...

    def check_ip(self, ip: str, per_minute: int, per_day: int) -> tuple[bool, int]: ...


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

    def take_tokens(self, name: str, n: int, per_minute: int | None, per_day: int | None) -> bool:
        """Reserve an estimated ``n`` tokens against the model's per-minute and per-day budgets. When either
        is over, the reservation is given back and False is returned, so the chain moves on before the
        provider would answer 429."""
        now = self.clock()
        keys = [
            (f"t:{name}:m:{int(now // 60)}", 90, per_minute),
            (f"t:{name}:d:{_day(now)}", 90000, per_day),
        ]
        totals = [self.hit(k, ttl, n) for k, ttl, _ in keys]
        self.last_block = ""
        if any(
            cap is not None and total > cap for total, (_, _, cap) in zip(totals, keys, strict=True)
        ):
            self.last_block = (
                "minute"
                if totals[0] > (per_minute if per_minute is not None else 10**12)
                else "day"
            )
            for k, ttl, _ in keys:
                self.hit(k, ttl, -n)
            return False
        return True

    def adjust_tokens(self, name: str, delta: int) -> None:
        """Replace an estimate by the real usage once the provider has reported it."""
        if delta:
            now = self.clock()
            self.hit(f"t:{name}:m:{int(now // 60)}", 90, delta)
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


def make_limiter(
    settings: AskSettings, clock=time.time, transport: httpx.BaseTransport | None = None
) -> Limiter:
    mem = MemoryLimiter(clock, settings.ip_salt)
    if settings.upstash_url and settings.upstash_token:
        return UpstashLimiter(
            settings.upstash_url, settings.upstash_token, mem, clock, settings.ip_salt, transport
        )
    return mem
