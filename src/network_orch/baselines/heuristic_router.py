"""Lexical keyword router (previous system, arm C). Baseline only; not used by the cascade."""

from __future__ import annotations

from typing import Literal

_QUERY_STARTS = (
    "what", "how", "are", "is", "can", "do", "does", "where", "which",
    "should", "check", "list", "describe", "summarize", "tell me",
)
_MIGRATE_VERBS = ("migrat", "shift", "move", "relocate")


def route_intent(intent: str) -> Literal["scale", "migrate", "monitor"]:
    t = intent.lower()
    if any(t.startswith(q) for q in _QUERY_STARTS):
        return "monitor"
    if any(v in t for v in _MIGRATE_VERBS):
        return "migrate"
    if any(k in t for k in ("replica", "scale", "load", "cpu", "traffic",
                             "capacity", "spin", "throughput", "burst",
                             "connection")):
        return "scale"
    if any(k in t for k in ("migrat", "edge-", "zone")):
        return "migrate"
    return "monitor"
