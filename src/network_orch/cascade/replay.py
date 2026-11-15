"""Replay T0: serves precomputed calibrated predictions keyed by intent text.

Lets the cascade (and real-LLM arms) run locally with encoder outputs that were
produced elsewhere (e.g. Kaggle), without loading any model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from network_orch.cascade.actions import ActionKind
from network_orch.cascade.decision import KINDS, Prediction


class ReplayDecisionModel:
    def __init__(self, texts: Sequence[str], kind_probs: Sequence[Sequence[float]],
                 p_unsafe: Sequence[float]) -> None:
        self._by_text: Mapping[str, Prediction] = {
            t: Prediction({k: float(p[i]) for i, k in enumerate(KINDS)}, float(u))
            for t, p, u in zip(texts, kind_probs, p_unsafe)
        }

    def predict(self, text: str) -> Prediction:
        try:
            return self._by_text[text]
        except KeyError:  # unseen text: maximally uncertain, forces escalation
            return Prediction({k: 1 / len(KINDS) for k in KINDS}, 0.5)

    def __len__(self) -> int:
        return len(self._by_text)


def from_scores(rows: Sequence[dict], route_scores: Sequence[Sequence[float]],
                unsafe_scores: Sequence[Sequence[float]]) -> ReplayDecisionModel:
    """rows: dataset rows of one split; unsafe_scores rows are [p_safe, p_unsafe]."""
    return ReplayDecisionModel([r["text"] for r in rows], route_scores,
                               [p[1] for p in unsafe_scores])


__all__ = ["ReplayDecisionModel", "from_scores", "ActionKind"]
