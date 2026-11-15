# Runbook: fault detection + self-healing (streamer)

Source: report §7.4 + §8.3.

## Detect
- Readiness/liveness failures, 100% packet loss on node, CrashLoopBackOff.

## Respond
1. Mark node unhealthy (cordon in full ops; not in v1 demo scope).
2. ReplicaSet self-heals stateless pods automatically — agent verifies, does NOT
   fight the controller (no duplicate scale during healing).
3. If zone is lost: `migrate_workload` survivors to healthy zone.
4. Notify: "mitigated autonomously, hardware repair still needs a human."

## Verify + learn
- All pods Ready ≤ 5 min; brief disruption budget ~5s stateless / ~30s stateful.
- Append incident to telemetry + RAG episodic memory; adjust thresholds.

## Scaling, migration, and recovery after a heal (cross-topic distractor)
- Once pods are Ready again, normal procedures resume: reactive scale when
  CPU > 80% for 3+ min (bounds 1..8), predictive scale before the evening
  peak, migration to another edge zone via `nodeSelector` (p95 back under
  50 ms), and canary+rollback for risky moves. These are references shared
  with the scaling/migration/recovery runbooks, not the healing procedure.
