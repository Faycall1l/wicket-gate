"""T0 decision model: calibrated action-kind + unsafe probabilities.

`LinearDecisionModel` is a thin, swappable baseline (pluggable featurizer +
logistic heads + temperature scaling fit on a held-out split). Heavier open
decision models (Laya/Von/Kev) implement the same `DecisionModel` protocol.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from network_orch.cascade.actions import ActionKind

KINDS = [ActionKind.SCALE, ActionKind.MIGRATE, ActionKind.MONITOR]


@dataclass(frozen=True)
class Prediction:
    probs: dict[ActionKind, float]
    p_unsafe: float
    latency_ms: float = 0.0

    @property
    def top(self) -> ActionKind:
        return max(self.probs, key=self.probs.get)

    @property
    def confidence(self) -> float:
        return self.probs[self.top]

    def prediction_set(self, mass: float = 0.9) -> list[ActionKind]:
        """Smallest set of kinds whose probability mass reaches `mass`."""
        out, total = [], 0.0
        for k, p in sorted(self.probs.items(), key=lambda kv: -kv[1]):
            out.append(k)
            total += p
            if total >= mass:
                break
        return out


class DecisionModel(Protocol):
    def predict(self, text: str) -> Prediction: ...


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(t: float) -> float:
        p = _softmax(logits / t)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean())
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


class LinearDecisionModel:
    def __init__(self, featurizer: Callable[[Sequence[str]], object] | None = None,
                 c: float = 10.0) -> None:
        self._tfidf: TfidfVectorizer | None = None
        self._featurize = featurizer
        self._c = c
        self._kind = LogisticRegression(max_iter=2000, C=c)
        self._unsafe = LogisticRegression(max_iter=2000, C=c)
        self.t_kind = self.t_unsafe = 1.0

    def _x(self, texts: Sequence[str]):
        return self._featurize(texts) if self._featurize else self._tfidf.transform(texts)

    def fit(self, texts: Sequence[str], kinds: Sequence[ActionKind],
            unsafe: Sequence[int], cal_texts: Sequence[str],
            cal_kinds: Sequence[ActionKind], cal_unsafe: Sequence[int]
            ) -> LinearDecisionModel:
        if self._featurize is None:
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts)
        yk = np.array([KINDS.index(k) for k in kinds])
        self._kind.fit(self._x(texts), yk)
        self._unsafe.fit(self._x(texts), np.array(unsafe))
        xc = self._x(cal_texts)
        self.t_kind = _fit_temperature(self._kind.decision_function(xc),
                                       np.array([KINDS.index(k) for k in cal_kinds]))
        zu = self._unsafe.decision_function(xc)
        self.t_unsafe = _fit_temperature(np.stack([-zu / 2, zu / 2], 1),
                                         np.array(cal_unsafe))
        return self

    def predict_batch(self, texts: Sequence[str]) -> list[Prediction]:
        t0 = time.perf_counter()
        x = self._x(texts)
        pk = _softmax(self._kind.decision_function(x) / self.t_kind)
        zu = self._unsafe.decision_function(x)
        pu = _softmax(np.stack([-zu / 2, zu / 2], 1) / self.t_unsafe)[:, 1]
        ms = (time.perf_counter() - t0) * 1e3 / max(len(texts), 1)
        return [Prediction({k: float(pk[i, j]) for j, k in enumerate(KINDS)},
                           float(pu[i]), ms) for i in range(len(texts))]

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> LinearDecisionModel:
        return joblib.load(path)
