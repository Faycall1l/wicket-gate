# Runbook: service migration with continuity (streamer)

Source: report §7.3 + §8.1. Tool contract: `migrate_workload(deployment, to_zone)`.

## Triggers
- Latency > 75% of 50ms SLA, user mobility away from serving edge, resource
  exhaustion, or better capacity elsewhere with benefit > cost.

## Mechanism (what the tool does)
- Patches Deployment pod-template `nodeSelector topology.network-orch/zone`.
- ReplicaSet reschedules; Service selector is UNCHANGED so traffic follows new pods.
- Stateless image (nginx) → start-new + redirect, no state transfer. Stateful
  services would need checkpoint/restore (see healing.md).

## Procedure
1. `get_cluster_state` → source zone, destination readiness + capacity.
2. Reject no-op (source == dest) and unknown zones (edge-a/b/c, cloud).
3. Dry-run diff → approval → execute → verify Ready on dest, p95 back < 50ms.
4. Log episodic memory: pattern, latency before/after, cost.

Continuity metric: success rate + p95 during reschedule window.

## Safety net
- Use `canary_migrate` when budget allows (health-gated, reversible).
- On any failure sign post-coordination: `rollback` (see recovery.md).
