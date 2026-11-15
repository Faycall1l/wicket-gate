"""Typed action space and decision record shared by every cascade tier."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ActionKind(str, Enum):
    SCALE = "scale"
    MIGRATE = "migrate"
    MONITOR = "monitor"  # read-only
    REFUSE = "refuse"    # unsafe or out of scope; nothing is executed


class Tier(str, Enum):
    CACHE = "cache"
    T0 = "t0"            # edge-local decision model
    T1 = "t1"            # LLM teacher
    HUMAN = "human"      # abstained: needs an operator


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    deployment: str = "streamer"
    namespace: str = "streaming-demo"
    replicas: int | None = None   # SCALE only
    to_zone: str | None = None    # MIGRATE only

    def tool_call(self) -> tuple[str, dict]:
        """Map to the MCP tool surface used by the executor."""
        if self.kind is ActionKind.SCALE:
            return "scale_workload", {
                "deployment": self.deployment, "namespace": self.namespace,
                "replicas": self.replicas}
        if self.kind is ActionKind.MIGRATE:
            return "migrate_workload", {
                "deployment": self.deployment, "namespace": self.namespace,
                "to_zone": self.to_zone}
        return "get_cluster_state", {}


@dataclass
class Decision:
    action: Action | None          # None = deferred past the last tier
    tier: Tier
    confidence: float
    latency_ms: float
    violations: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)  # tier-by-tier audit notes
