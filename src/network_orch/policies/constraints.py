"""MeshAgent-style invariants: validate every proposed write before it executes.

Each validator returns a list of violations (empty = valid). The orchestrator
rejects invalid proposals and the MCP server re-checks independently, so a
prompt-injection or bad LLM output cannot bypass safety by calling tools directly.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_REPLICAS = 1
MAX_REPLICAS = 8
VALID_ZONES = ("edge-a", "edge-b", "edge-c", "cloud")
PROTECTED_NAMESPACES = ("kube-system", "kube-public", "kube-node-lease")


@dataclass(frozen=True)
class ScaleProposal:
    namespace: str
    deployment: str
    replicas: int


@dataclass(frozen=True)
class MigrateProposal:
    namespace: str
    deployment: str
    from_zone: str
    to_zone: str


def validate_scale(p: ScaleProposal) -> list[str]:
    violations: list[str] = []
    if p.namespace in PROTECTED_NAMESPACES:
        violations.append(f"namespace {p.namespace} is protected")
    if not (MIN_REPLICAS <= p.replicas <= MAX_REPLICAS):
        violations.append(
            f"replicas {p.replicas} outside bounds [{MIN_REPLICAS},{MAX_REPLICAS}]"
        )
    if not p.deployment:
        violations.append("deployment name is empty")
    return violations


def validate_migrate(p: MigrateProposal) -> list[str]:
    violations: list[str] = []
    if p.namespace in PROTECTED_NAMESPACES:
        violations.append(f"namespace {p.namespace} is protected")
    if p.to_zone not in VALID_ZONES:
        violations.append(f"unknown destination zone {p.to_zone}")
    if p.from_zone == p.to_zone:
        violations.append("no-op migration: source == destination")
    if not p.deployment:
        violations.append("deployment name is empty")
    return violations


def confidence_ok(confidence: float, threshold: float) -> bool:
    """Abstain when the proposer is not confident enough (MeshAgent abstention)."""
    return confidence >= threshold


def check_tool_constraints(tool: str, args: dict,
                           current_zone: str = "?") -> list[str]:
    """Shared constraint core (R2): tool-arg validation with NO abstain
    gate and NO cluster lookup. Namespace comes from args (default
    streaming-demo, issue #52); the caller resolves current_zone however
    it can (dry-run backend, helper, or "?" unknown). The graph validator
    node and the offline mirrors (calib_sweep, latency_probe) all delegate
    here so the rule cannot drift between them."""
    args = args or {}
    ns = args.get("namespace") or "streaming-demo"
    if tool == "scale_workload":
        return validate_scale(ScaleProposal(
            ns, args.get("deployment", ""), int(args.get("replicas", 0))))
    if tool == "migrate_workload":
        return validate_migrate(MigrateProposal(
            ns, args.get("deployment", ""), current_zone,
            args.get("to_zone", "")))
    return []  # monitor / unknown tools stay read-only


# --- SLO-aware demand signals (issue #44, KEDA queue-depth pattern) ---
# Additive only: existing validators are untouched. A ScaleDemand carries the
# demand evidence behind a scale proposal; the bounds encode asymmetric
# behavior (fast up, slow down) and gate scale-to-zero to dev namespaces
# (cold-start caveat: GPU/model load takes minutes — prod keeps a floor).

# Max replicas added in a single scale-up decision (fast-up cap).
MAX_STEP_UP = 4
# Max replicas removed in a single scale-down decision (slow-down floor).
MAX_STEP_DOWN = 1
# Queue depth per replica above which scale-up is warranted.
QUEUE_PER_REPLICA = 5
# Namespaces where scale-to-zero is permitted (dev only).
ZERO_OK_NAMESPACES = ("dev-demo",)


@dataclass(frozen=True)
class ScaleDemand:
    namespace: str
    deployment: str
    current: int
    target: int
    queue_depth: int = 0
    p95_ms: float = 0.0


def validate_scale_demand(p: ScaleDemand) -> list[str]:
    """Validate a demand-evidenced scale step. Empty = valid."""
    violations: list[str] = []
    if p.namespace in PROTECTED_NAMESPACES:
        violations.append(f"namespace {p.namespace} is protected")
    if not (MIN_REPLICAS <= p.target <= MAX_REPLICAS) and p.target != 0:
        violations.append(
            f"replicas {p.target} outside bounds [{MIN_REPLICAS},{MAX_REPLICAS}]"
        )
    if p.target == 0 and p.namespace not in ZERO_OK_NAMESPACES:
        violations.append(f"scale-to-zero only in {list(ZERO_OK_NAMESPACES)}")
    step = p.target - p.current
    if step > MAX_STEP_UP:
        violations.append(f"scale-up step {step} exceeds cap {MAX_STEP_UP}")
    if -step > MAX_STEP_DOWN:
        violations.append(f"scale-down step {-step} exceeds floor {MAX_STEP_DOWN}")
    if step > 0 and p.queue_depth < QUEUE_PER_REPLICA * p.current:
        violations.append(
            f"queue_depth {p.queue_depth} below demand bar "
            f"{QUEUE_PER_REPLICA * p.current} for scale-up"
        )
    if not p.deployment:
        violations.append("deployment name is empty")
    return violations
