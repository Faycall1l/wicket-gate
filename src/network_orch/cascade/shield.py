"""Hard safety shield: the validator is a constraint, never a learned policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from network_orch.cascade.actions import Action, ActionKind
from network_orch.policies.constraints import check_tool_constraints


@dataclass(frozen=True)
class Verdict:
    ok: bool
    incomplete: bool = False  # missing parameters: defer upward, don't refuse
    violations: list[str] = field(default_factory=list)


def check(action: Action, current_zone: str = "?") -> Verdict:
    """Validate one action against the committed constraint core."""
    if action.kind is ActionKind.REFUSE:
        return Verdict(ok=False, violations=["refused by policy"])
    if action.kind is ActionKind.SCALE and action.replicas is None:
        return Verdict(ok=False, incomplete=True, violations=["replicas missing"])
    if action.kind is ActionKind.MIGRATE:
        if not action.to_zone:
            return Verdict(ok=False, incomplete=True, violations=["to_zone missing"])
        if current_zone == "?":  # nobody resolved the source: not unsafe, not applicable
            return Verdict(ok=False, incomplete=True,
                           violations=["source zone unresolved"])
    tool, args = action.tool_call()
    violations = check_tool_constraints(tool, args, current_zone=current_zone)
    return Verdict(ok=not violations, violations=violations)
