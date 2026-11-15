"""Verified-decision cache: only shield-approved, outcome-verified actions."""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass

from network_orch.cascade.actions import Action

_NOISE = re.compile(r"\bticket\s*#?\d+\b|[^\w\s-]")


def normalize(text: str) -> str:
    """Lowercase, drop ticket ids and punctuation, collapse whitespace.

    Numbers other than ticket ids are kept: "to 100" and "to 200" must differ.
    """
    return " ".join(_NOISE.sub(" ", text.lower()).split())


@dataclass
class _Entry:
    action: Action
    stored_at: float
    hits: int = 0


class VerifiedDecisionCache:
    """LRU + TTL map from normalized intent text to a verified Action."""

    def __init__(self, max_size: int = 4096, ttl_s: float = 3600.0) -> None:
        self._d: OrderedDict[str, _Entry] = OrderedDict()
        self.max_size, self.ttl_s = max_size, ttl_s
        self.lookups = self.hits = 0

    def get(self, text: str, now: float | None = None) -> Action | None:
        self.lookups += 1
        key, now = normalize(text), now if now is not None else time.time()
        e = self._d.get(key)
        if e is None or now - e.stored_at > self.ttl_s:
            self._d.pop(key, None)
            return None
        self._d.move_to_end(key)
        e.hits += 1
        self.hits += 1
        return e.action

    def put(self, text: str, action: Action, now: float | None = None) -> None:
        key = normalize(text)
        self._d[key] = _Entry(action, now if now is not None else time.time())
        self._d.move_to_end(key)
        while len(self._d) > self.max_size:
            self._d.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def __len__(self) -> int:
        return len(self._d)
