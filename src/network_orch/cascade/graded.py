"""Graded verdicts between an agent's proposal and the cluster.

ALLOW     apply the action (a same-zone migration is a harmless no-op)
REPAIR    return the reason to the agent and ask for a corrected proposal (bounded retries)
ASK       parameters are missing or ambiguous: ask the operator
ESCALATE  valid but consequential (scale to zero): needs approval
DENY      hard invariant (protected namespace) or the agent refused

The rules are deterministic and reuse the constants of the strict validator, so the guard never
admits anything the validator rejects as a hard invariant. It differs from the validator in what
happens to a rejected proposal: it is routed (repair, ask, escalate) instead of refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from network_orch.cascade.actions import Action, ActionKind
from network_orch.cascade import slots
from network_orch.policies.constraints import (MAX_REPLICAS, MIN_REPLICAS, PROTECTED_NAMESPACES, VALID_ZONES)

ALLOW, REPAIR, ASK, ESCALATE, DENY = "ALLOW", "REPAIR", "ASK", "ESCALATE", "DENY"


@dataclass(frozen=True)
class Policy:
    """Limits the checks enforce. Defaults match the evaluated configuration.

    `namespaces`, when set, is an allowlist: a proposal for any other namespace is sent back for repair.
    It is off by default because the published results were measured without it.
    """
    min_replicas: int = MIN_REPLICAS
    max_replicas: int = MAX_REPLICAS
    zones: tuple[str, ...] = tuple(VALID_ZONES)
    protected_namespaces: tuple[str, ...] = tuple(PROTECTED_NAMESPACES)
    namespaces: Optional[tuple[str, ...]] = field(default=None)

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        known = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


DEFAULT = Policy()


@dataclass(frozen=True)
class Decision:
    verdict: str
    reason: str = ""
    action: Optional[Action] = None  # the action to apply when verdict == ALLOW
    retries: int = 0


def decide(action: Action, underspecified: str = "", policy: Policy = DEFAULT) -> tuple[str, str]:
    """Verdict and reason for one proposal. `underspecified` is the reason from `slots.underspecified(text)`."""
    if underspecified:
        return ASK, underspecified
    k = action.kind
    if k is ActionKind.MONITOR:
        return ALLOW, ""
    if k is ActionKind.REFUSE:
        return DENY, "agent refused"
    if action.namespace in policy.protected_namespaces:
        return DENY, f"namespace {action.namespace} is protected"
    if policy.namespaces is not None and action.namespace not in policy.namespaces:
        return REPAIR, f"namespace {action.namespace} is not managed; allowed namespaces are {', '.join(policy.namespaces)}"
    if k is ActionKind.SCALE:
        if action.replicas is None:
            return ASK, "replica count missing"
        if action.replicas == 0:
            return ESCALATE, "scale to zero needs approval"
        if not (policy.min_replicas <= action.replicas <= policy.max_replicas):
            return REPAIR, f"replicas {action.replicas} are outside the allowed range [{policy.min_replicas},{policy.max_replicas}]"
        return ALLOW, ""
    if not action.to_zone:
        return ASK, "destination zone missing"
    if action.to_zone not in policy.zones:
        return REPAIR, f"zone {action.to_zone} does not exist; valid zones are {', '.join(policy.zones)}"
    return ALLOW, ""


def describe(a: Action) -> str:
    return f"{a.kind.value} replicas={a.replicas} to_zone={a.to_zone} namespace={a.namespace}"


def guard(text: str, propose: Callable[[str], Optional[Action]], retries: int = 2, policy: Policy = DEFAULT) -> Decision:
    """Run the agent (`propose(text) -> Action | None`) behind the graded checks with a bounded repair loop."""
    why = slots.underspecified(text)
    act = propose(text)
    if act is None:
        return Decision(DENY, "no proposal")
    verdict, reason = decide(act, why, policy)
    tries = 0
    while verdict == REPAIR and tries < retries:
        tries += 1
        act = propose(f"{text}\n\nA reviewer rejected your previous proposal ({describe(act)}): {reason}. "
                      "Propose a corrected action, or refuse if the request cannot be met within the limits.")
        if act is None:
            return Decision(DENY, "no valid retry", retries=tries)
        verdict, reason = decide(act, why, policy)
    return Decision(verdict, reason, act if verdict == ALLOW else None, tries)
