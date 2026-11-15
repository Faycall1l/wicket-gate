"""Risk-controlled gate: act on T0 only when certified unsafe-apply risk <= alpha.

Calibration (split / learn-then-test style): on a held-out calibration set pick
the largest threshold tau such that the Clopper-Pearson upper bound on the
unsafe fraction among *allowed* items is <= alpha. At serve time T0's P(unsafe)
must be < tau to act. `AdaptiveGate` additionally tracks realized outcomes and
tightens/loosens tau online so the bound survives workload drift (ACI-style).
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import beta


def cp_upper(k: int, n: int, delta: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound for k unsafe among n allowed."""
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


@dataclass
class Certificate:
    alpha: float
    tau: float | None          # None => risk target not certifiable
    n_allowed: int = 0
    n_unsafe_allowed: int = 0
    allow_rate: float = 0.0
    upper_bound: float = 1.0

    @property
    def certified(self) -> bool:
        return self.tau is not None


def calibrate(p_unsafe: list[float], is_unsafe: list[int], alpha: float,
              delta: float = 0.05, grid: int = 200) -> Certificate:
    """Largest tau with CP-upper(unsafe | allowed) <= alpha; else uncertified."""
    n_total = len(p_unsafe)
    best = Certificate(alpha=alpha, tau=None)
    for i in range(1, grid + 1):
        tau = 0.5 * i / grid
        allowed = [u for p, u in zip(p_unsafe, is_unsafe) if p < tau]
        n, k = len(allowed), sum(allowed)
        if n == 0:
            continue
        ub = cp_upper(k, n, delta)
        if ub <= alpha:
            best = Certificate(alpha, tau, n, k, n / n_total, ub)
    return best


def calibrate_worst_group(p_unsafe: list[float], is_unsafe: list[int],
                          groups: list[str], alpha: float, delta: float = 0.05,
                          grid: int = 200, min_group_allowed: int = 8) -> Certificate:
    """Shift-aware calibration: pooled CP bound AND every group's empirical risk
    among allowed items must be <= alpha (groups = prompt templates / sources).

    Groups with fewer than `min_group_allowed` allowed items are skipped at that
    tau; this is a worst-group heuristic, not a guarantee for unseen groups."""
    n_total = len(p_unsafe)
    best = Certificate(alpha=alpha, tau=None)
    for i in range(1, grid + 1):
        tau = 0.5 * i / grid
        idx = [j for j, p in enumerate(p_unsafe) if p < tau]
        if not idx:
            continue
        k = sum(is_unsafe[j] for j in idx)
        ub = cp_upper(k, len(idx), delta)
        if ub > alpha:
            continue
        by_group: dict[str, list[int]] = {}
        for j in idx:
            by_group.setdefault(groups[j], []).append(is_unsafe[j])
        worst = max((sum(v) / len(v) for v in by_group.values()
                     if len(v) >= min_group_allowed), default=0.0)
        if worst <= alpha:
            best = Certificate(alpha, tau, len(idx), k, len(idx) / n_total, ub)
    return best


class AdaptiveGate:
    """Serve-time gate with online threshold adaptation under drift.

    After each acted decision the realized outcome (unsafe or not) updates
    tau with an additive step: a miss shrinks tau by `gamma*(1-alpha)`, a
    success grows it by `gamma*alpha`, so long-run miss rate tracks alpha.
    """

    def __init__(self, cert: Certificate, gamma: float = 0.005,
                 tau_max: float = 0.5) -> None:
        if not cert.certified:
            raise ValueError("cannot serve an uncertified gate")
        self.alpha = cert.alpha
        self.tau = cert.tau
        self.gamma = gamma
        self.tau_max = tau_max

    def allow(self, p_unsafe: float) -> bool:
        return p_unsafe < self.tau

    def observe(self, was_unsafe: bool) -> None:
        step = -self.gamma * (1 - self.alpha) if was_unsafe else self.gamma * self.alpha
        self.tau = min(self.tau_max, max(0.0, self.tau + step))
