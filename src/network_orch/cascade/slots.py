"""Parameter extraction: pulls explicit replicas / zone out of an intent.

Deliberately conservative: anything not stated explicitly returns None so the
cascade defers to the LLM instead of guessing a number or zone.
"""

from __future__ import annotations

import re

from network_orch.policies.constraints import PROTECTED_NAMESPACES, VALID_ZONES

_REPLICAS = re.compile(
    r"\b(?:to|run|need|give it|set(?: to)?|with)\s+(-?\d+)\s*"
    r"(?:replicas?|pods?|instances?|copies)?\b|\b(-?\d+)\s*(?:replicas?|pods?|instances?)\b")
_ZONE = re.compile(r"\b(edge-[a-z]|cloud)\b")


def replicas(text: str) -> int | None:
    m = _REPLICAS.search(text.lower())
    if not m:
        return None
    return int(next(g for g in m.groups() if g is not None))


_VAGUE = {"nearest", "closest", "another", "other", "somewhere", "it", "there", "best"}
_DEST = re.compile(r"(?:\bto|->|\bonto|\binto)\s+(?:the\s+)?([a-z][\w-]*)")


def destination_zone(text: str) -> str | None:
    """Destination named after to/->/onto/into, verbatim (so unknown zones such as
    'mars' reach the shield instead of being silently replaced); None if vague."""
    t = text.lower()
    for m in _DEST.finditer(t):
        tok = m.group(1)
        if tok in _VAGUE:
            return None
        if tok.startswith("edge-") or tok == "cloud" or tok in {"mars", "moon"} \
                or re.fullmatch(r"[a-z]+-[a-z0-9]+", tok):
            return tok
    zones = _ZONE.findall(t)
    return zones[-1] if zones else None


def source_zone(text: str) -> str | None:
    """Zone the workload currently sits in: a zone mention that is not the
    destination occurrence (a no-op migration repeats the same zone)."""
    t = text.lower()
    dest = _DEST.search(t)
    spans = [m for m in _ZONE.finditer(t)]
    if dest:
        spans = [m for m in spans if not (dest.start(1) <= m.start() < dest.end(1))]
    return spans[0].group(1) if spans else None


_MULTIPLIER = re.compile(
    r"\b\d+\s*x\b|\b(?:double|twice|treble|triple|quadruple)\b")


def underspecified(text: str) -> str | None:
    """Intents unresolvable to a concrete action from the text alone.

    Multiplier phrasing ("10x current", "triple then double it") carries no
    absolute target: the shield cannot validate it, and any concrete number a
    teacher emits could be quietly illegal-or-wrong. Such intents must reach a
    human, never be auto-applied. (Source-zone-less migrates are handled by
    the shield's incomplete verdict instead.)
    """
    if _MULTIPLIER.search(text.lower()):
        return "multiplier phrasing with no absolute target"
    return None


def known_zone(zone: str | None) -> bool:
    return zone in VALID_ZONES


def namespace(text: str) -> str:
    """Explicitly named protected namespace, else the default."""
    t = text.lower()
    return next((ns for ns in PROTECTED_NAMESPACES if ns in t), "streaming-demo")
