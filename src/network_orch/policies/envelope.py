"""Typed-policy envelope + arbitration (issue #41, AURA/A1gent pattern).

Enforced in the executor exactly when the validator backstop is on
(`enforce` flag; raw splat under enforce=False for ablation fidelity).
A TypedPolicy carries action + bounded params + guardrail refs in one
schema-validated object; arbitration picks a single winner
deterministically when proposals interact, with dwell-time stickiness
and a confidence deadband so near-ties do not flap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from network_orch.policies.constraints import (
    MigrateProposal,
    ScaleProposal,
    validate_migrate,
    validate_scale,
)

KNOWN_ACTIONS = ("get_cluster_state", "scale_workload", "migrate_workload")
# Lower acts first: reads < replica-count changes < traffic moves.
BLAST_RADIUS = {"get_cluster_state": 0, "scale_workload": 1, "migrate_workload": 2}


@dataclass(frozen=True)
class TypedPolicy:
    action: str
    namespace: str
    deployment: str
    params: dict = field(default_factory=dict)
    confidence: float = 0.0
    guardrails: tuple = ()


def validate_envelope(p: TypedPolicy) -> list[str]:
    """Schema + bounds check. Empty = valid and executable-in-principle."""
    violations: list[str] = []
    if p.action not in KNOWN_ACTIONS:
        return [f"unknown action {p.action}"]
    if not p.guardrails:
        violations.append("envelope carries no guardrail refs")
    if not (0.0 <= p.confidence <= 1.0):
        violations.append(f"confidence {p.confidence} outside [0,1]")
    if p.action == "scale_workload":
        violations += validate_scale(ScaleProposal(
            p.namespace, p.deployment, int(p.params.get("replicas", 0))))
    elif p.action == "migrate_workload":
        violations += validate_migrate(MigrateProposal(
            p.namespace, p.deployment, p.params.get("from_zone", "?"),
            p.params.get("to_zone", "")))
    return violations


def arbitrate(policies: list[TypedPolicy], incumbent: TypedPolicy | None = None,
              incumbent_at: float | None = None, now: float = 0.0,
              dwell_s: float = 300.0, deadband: float = 0.05
              ) -> tuple[TypedPolicy | None, str, list[str]]:
    """Pick one winner deterministically.

    Returns (winner|None, reason, log). Rules, in order:
    1. invalid envelopes are dropped (each logged);
    2. highest confidence wins; ties break on lower blast radius, then
       list order (stable);
    3. dwell: a live incumbent keeps winning inside dwell_s;
    4. deadband: a challenger within `deadband` confidence of the incumbent
       with >= blast radius does not displace it (no flapping on near-ties
       toward higher-blast actions).
    """
    log: list[str] = []
    valid = []
    for p in policies:
        v = validate_envelope(p)
        if v:
            log.append(f"drop {p.action}/{p.deployment}: {'; '.join(v)}")
        else:
            valid.append(p)
    if not valid:
        return None, "no valid policies", log
    ordered = sorted(valid, key=lambda p: (-p.confidence, BLAST_RADIUS[p.action]))
    top = ordered[0]
    if incumbent is not None and validate_envelope(incumbent):
        log.append("incumbent invalid; re-arbitrating")
        incumbent = None
    if incumbent is not None:
        if incumbent_at is not None and now - incumbent_at < dwell_s:
            return incumbent, f"dwell ({now - incumbent_at:.0f}s < {dwell_s:.0f}s)", log
        for c in ordered:
            if (abs(c.confidence - incumbent.confidence) <= deadband
                    and BLAST_RADIUS[c.action] >= BLAST_RADIUS[incumbent.action]
                    and (c.action, c.deployment) != (incumbent.action, incumbent.deployment)):
                log.append(f"deadband holds {incumbent.action} over {c.action}")
                return incumbent, "deadband", log
    log.append(f"elect {top.action}/{top.deployment}")
    return top, "elect", log
