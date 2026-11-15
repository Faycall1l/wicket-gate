# Runbook: reactive + predictive scaling (streamer)

Source: report §7.2. Tool contract: `scale_workload(deployment, replicas)`.

## When to scale up
- CPU > 80% for 3+ min, or p95 latency rising with queue depth, or forecasted
  spike (lunch 12-14h / evening 19-22h patterns).
- Bounds enforced: 1..8 replicas (see policies/constraints.py).

## Procedure
1. `get_cluster_state` → confirm current replicas + node readiness.
2. `query_runbook("scaling bounds streamer")` → cite this file.
3. Propose `scale_workload(streamer, N+2)` on spike, `N+1` on steady pressure.
4. Validate (bounds, protected namespace) → dry-run diff → approval → execute.
5. Verify: Deployment `ready == replicas` within 120s; rollback on ImagePull/unschedulable.

## Scale down
- CPU < 20% for 10+ min → -1, never below 1. Gradual, verify between steps.
