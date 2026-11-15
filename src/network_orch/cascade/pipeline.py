"""Cascade orchestration: cache -> T0 -> gate -> shield -> T1 -> human.

Every decision records its tier, latency, and an audit trace. Actions that
pass the shield are cached and appended to `learn_buffer` (verified rows for
later distillation of T0).
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from network_orch.cascade import shield, slots
from network_orch.cascade.actions import Action, ActionKind, Decision, Tier
from network_orch.cascade.cache import VerifiedDecisionCache
from network_orch.cascade.decision import DecisionModel
from network_orch.cascade.gate import AdaptiveGate
from network_orch.cascade.teacher import Teacher

ZoneFn = Callable[[], str]


@dataclass
class CascadeConfig:
    refuse_above: float = 0.5   # P(unsafe) at/above this: refuse without the LLM
    min_confidence: float = 0.8  # T0 top-kind probability needed to act alone
    set_mass: float = 0.9        # coverage of the prediction set sent to T1


@dataclass
class Cascade:
    t0: DecisionModel
    gate: AdaptiveGate
    t1: Teacher | None = None
    cache: VerifiedDecisionCache = field(default_factory=VerifiedDecisionCache)
    cfg: CascadeConfig = field(default_factory=CascadeConfig)
    current_zone: ZoneFn = lambda: "?"
    learn_buffer: list[tuple[str, Action]] = field(default_factory=list)
    tier_counts: Counter = field(default_factory=Counter)

    def decide(self, text: str) -> Decision:
        t_start = time.perf_counter()
        d = self._decide(text)
        d.latency_ms = (time.perf_counter() - t_start) * 1e3
        self.tier_counts[d.tier] += 1
        return d

    # -- internals -------------------------------------------------------
    def _accept(self, text: str, action: Action, tier: Tier, conf: float,
                trace: list[str], learn: bool = True) -> Decision:
        if learn and action.kind is not ActionKind.REFUSE:
            self.cache.put(text, action)
            self.learn_buffer.append((text, action))
        return Decision(action, tier, conf, 0.0, trace=trace)

    def _decide(self, text: str) -> Decision:
        why = slots.underspecified(text)
        if why is not None:  # "10x current": never auto-apply, never even gate it
            return Decision(Action(ActionKind.REFUSE), Tier.T0, 1.0, 0.0,
                            [f"underspecified: {why}"], trace=[f"refused: {why}"])
        zone = self.current_zone()
        if zone == "?":  # no cluster lookup: fall back to what the text states
            zone = slots.source_zone(text) or "?"
        trace: list[str] = []

        hit = self.cache.get(text)
        if hit is not None and shield.check(hit, zone).ok:
            return Decision(hit, Tier.CACHE, 1.0, 0.0, trace=["cache hit"])

        pred = self.t0.predict(text)
        trace.append(f"t0 top={pred.top.value} p={pred.confidence:.2f} "
                     f"unsafe={pred.p_unsafe:.2f}")
        if pred.p_unsafe >= self.cfg.refuse_above:
            trace.append("refused: high unsafe probability")
            return Decision(Action(ActionKind.REFUSE), Tier.T0,
                            pred.p_unsafe, 0.0, trace=trace)

        options = pred.prediction_set(self.cfg.set_mass)
        confident = (self.gate.allow(pred.p_unsafe) and len(options) == 1
                     and pred.confidence >= self.cfg.min_confidence)
        if confident:
            action = self._fill(pred.top, text)
            v = shield.check(action, zone)
            if v.ok:
                return self._accept(text, action, Tier.T0, pred.confidence, trace)
            if not v.incomplete:  # learned model proposed a constraint violation
                trace.append(f"shield blocked: {v.violations}")
                return Decision(Action(ActionKind.REFUSE), Tier.T0,
                                pred.confidence, 0.0, v.violations, trace)
            trace.append(f"t0 incomplete: {v.violations}")

        return self._escalate(text, options, zone, trace)

    def _fill(self, kind: ActionKind, text: str) -> Action:
        if kind is ActionKind.SCALE:
            return Action(kind, namespace=slots.namespace(text),
                          replicas=slots.replicas(text))
        if kind is ActionKind.MIGRATE:
            return Action(kind, namespace=slots.namespace(text),
                          to_zone=slots.destination_zone(text))
        return Action(kind)

    def _escalate(self, text: str, options, zone: str, trace: list[str]) -> Decision:
        if self.t1 is None:
            trace.append("no teacher configured: abstain")
            return Decision(None, Tier.HUMAN, 0.0, 0.0, trace=trace)
        prop = self.t1.propose(text, options)
        if prop is None:
            trace.append("t1 unparseable: abstain")
            return Decision(None, Tier.HUMAN, 0.0, 0.0, trace=trace)
        v = shield.check(prop.action, zone)
        trace.append(f"t1 {prop.action.kind.value} conf={prop.confidence:.2f}")
        if prop.action.kind is ActionKind.REFUSE:
            return Decision(prop.action, Tier.T1, prop.confidence, 0.0, trace=trace)
        if not v.ok:
            trace.append(f"shield blocked t1: {v.violations}")
            return Decision(Action(ActionKind.REFUSE), Tier.T1, prop.confidence,
                            0.0, v.violations, trace)
        return self._accept(text, prop.action, Tier.T1, prop.confidence, trace)

    def record_outcome(self, was_unsafe: bool) -> None:
        """Feed a realized outcome back so the gate adapts to drift."""
        self.gate.observe(was_unsafe)
