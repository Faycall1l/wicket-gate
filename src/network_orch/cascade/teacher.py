"""T1: LLM teacher with a prompt narrowed to T0's prediction set.

Instead of open-ended generation the LLM chooses among the few competing
action kinds T0 could not separate, grounded only in runbook snippets for
those kinds. Fewer options and a short JSON reply cut decode time (the
dominant latency leg) and off-topic errors.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from network_orch.cascade.actions import Action, ActionKind

Chat = Callable[[str, str], str]  # (system, user) -> reply text
Retriever = Callable[[str, ActionKind], Sequence[str]]

_SYSTEM = (
    "You are an edge orchestration planner. Choose exactly one option from the "
    "list and reply with ONE JSON object: "
    '{"kind": <option>, "replicas": int|null, "to_zone": '
    '"edge-a"|"edge-b"|"edge-c"|"cloud"|null, "namespace": str|null, '
    '"confidence": 0-1}. namespace is a Kubernetes namespace only if the '
    'request names one, otherwise null. '
    'Use "refuse" if the request is unsafe or none of the options fit. '
    "Do not invent parameters the request does not imply; use null."
)


@dataclass(frozen=True)
class Proposal:
    action: Action
    confidence: float


def build_prompt(text: str, options: Sequence[ActionKind],
                 retrieve: Retriever | None = None, per_option: int = 2) -> str:
    lines = [f"Request: {text}", "Options: " + ", ".join(
        [k.value for k in options] + [ActionKind.REFUSE.value])]
    if retrieve:
        for k in options:
            for snip in list(retrieve(text, k))[:per_option]:
                lines.append(f"[{k.value}] {snip.strip()[:400]}")
    return "\n".join(lines)


def _salvage(blob: str) -> dict:
    """Field-wise recovery from near-JSON (e.g. a stray quote after a number)."""
    def grab(key: str, pat: str):
        m = re.search(rf'"{key}"\s*:\s*{pat}', blob)
        return m.group(1) if m else None
    rep = grab("replicas", r"(-?\d+)")
    conf = grab("confidence", r"([0-9.]+)")
    return {"kind": grab("kind", r'"([a-z]+)"'),
            "replicas": int(rep) if rep is not None else None,
            "to_zone": grab("to_zone", r'"([a-z-]+)"'),
            "namespace": grab("namespace", r'"([\w-]+)"'),
            "confidence": float(conf.rstrip(".")) if conf else 0.5}


def parse_reply(reply: str, allowed: Sequence[ActionKind]) -> Proposal | None:
    """Tolerant JSON parse; reject kinds outside the narrowed option set."""
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        d = _salvage(m.group(0))
    try:
        kind = ActionKind(d.get("kind"))
    except (ValueError, TypeError):
        return None
    if kind not in set(allowed) | {ActionKind.REFUSE}:
        return None
    replicas = d.get("replicas")
    return Proposal(
        Action(kind, replicas=int(replicas) if isinstance(replicas, (int, float)) else None,
               to_zone=d.get("to_zone") or None,
               namespace=d.get("namespace") or "streaming-demo"),
        float(d.get("confidence", 0.5)))


ALL_KINDS = [ActionKind.SCALE, ActionKind.MIGRATE, ActionKind.MONITOR]


class Teacher:
    """Narrowed first, widened once if the reply falls outside the narrowed set
    (T0 can be confidently wrong, so its prediction set may exclude the truth)."""

    def __init__(self, chat: Chat, retrieve: Retriever | None = None) -> None:
        self._chat, self._retrieve = chat, retrieve
        self.calls = 0

    def _ask(self, text: str, options: Sequence[ActionKind]) -> Proposal | None:
        self.calls += 1
        reply = self._chat(_SYSTEM, build_prompt(text, options, self._retrieve))
        return parse_reply(reply, options)

    def propose(self, text: str, options: Sequence[ActionKind]) -> Proposal | None:
        prop = self._ask(text, options)
        if prop is None and set(options) != set(ALL_KINDS):
            prop = self._ask(text, ALL_KINDS)
        return prop
