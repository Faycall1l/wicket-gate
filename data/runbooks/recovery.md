# Runbook: migration recovery (canary + rollback)

Source: H6 recovery scenario. Tool contract: `canary_migrate(deployment,
to_zone, step_size)` and `rollback(deployment, last_known_good)`.

## Canary (make-before-break)
- Shadow Deployment `{name}-canary` is cloned from the original with the pod
  nodeSelector retargeted; the Service selector (app label) is shared, so new
  pods serve traffic alongside old pods.
- One `step_size` of capacity is added to the canary and health-gated
  (`ready_replicas >= target`) BEFORE any old pod is removed. A failed gate
  aborts the move with the original untouched.

## Recovery criteria (condition -> do)
- Any canary health gate fails (pod CrashLoopBackOff, zone capacity, image
  pull): `rollback(workload, last_known_good)` and keep the original running.
- Original does not come Ready on the target after retarget (unhealthy
  workload lands on destination): `rollback` to last-known-good zone.
- Post-migration p95 > SLA for > 30 s after a successful switch:
  `rollback`; the last-known-good zone and replica count are restored from the
  Deployment annotations written at migration start, then the canary shadow is
  removed so only the recovered pods serve.

## Benefits over atomic-swap migration
- `migrate_workload` (nodeSelector patch) is reversible but only via a second
  patch and with no intermediate safety step. Canary+rollback gives a
  measurable recovery path with continuity during the rollback window, which
  is the central honesty check for "autonomy without governance is unsafe".

## Scaling and zone moves during an incident (cross-topic distractor)
- When CPU stays above 80% for 3+ minutes mid-recovery, the operator can
  still `scale_workload(streamer, N+2)` (bounds 1..8 enforced) or
  `migrate_workload(streamer, to_zone)` when resource exhaustion caused the
  failure; the `nodeSelector` patch keeps the Service selector unchanged, so
  traffic follows the new pods and p95 must return under 50 ms.
- The rules here are about sequencing a scale/migrate next to a rollback,
  NOT the steady-state scaling/migration procedures themselves.