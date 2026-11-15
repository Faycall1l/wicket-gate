# k6 continuity notes (issue #1)

## In-cluster generator (2026-09-15, issue #1) — standard harness path

`run_incluster.py` schedules `continuity.js` as a Job inside the cluster against
`svc/streamer` (no host<->cluster port-forward in the data path). The container
streams `raw.jsonl` to stdout (`sh -c "...; cat /tmp/raw.jsonl"`), so the raw
metric stream is harvested from pod logs even after the pod completes; summary
JSON is re-extracted from the same stdout. Artifacts land in
`out/incluster-<ts>/{k6-stdout.txt,summary.json,scored.json}` and `scored.json`
derives successes from `http_reqs[expected_response=true]` points.

| Run | Requests | Success | p50 | p95 | p99 |
|---|---|---|---|---|---|
| 30 s | 600 | 100% | 2.8 ms | 39.2 ms | 178 ms |
| 90 s (earlier pass) | 1,542 | 100% | 7.5 ms | 468.9 ms | 1,710 ms |
| 50 rps stress (30 s) | 1,453 | 100% | n/a | 13.8 ms | n/a |

Both runs cross-check the prober-py band (p50 ~1–9 ms) at the same 20 rps
design load; p95/p99 are elevated by the co-tenant baleen-neo4j host, matching
the loaded-host behavior seen in the stateful migration sweep. Success is 100%
on the data plane.

## pf-artifact attribution (historical, removed runs)

The 2026-09-13 migration-window runs put k6 behind `kubectl port-forward` and
showed an external "blackhole" once the reschedule started (success ~0%,
0 ms). Root cause: `kubectl port-forward` exits when the API-server proxy's
upstream connection to a terminating pod resets, closing the listen port, so
every later client sees connection-refused at 0 ms — a channel artifact, not
endpoint starvation (the in-cluster prober held 99.97% across the same
window). Those pf runs were non-measurements and are removed from the repo;
the data-plane claim uses the in-cluster channel only.

## Gotchas (k6 v2.2)

1. **Never pass `--duration` with `scenarios:`** — the CLI flag overrides and
   disables scenarios entirely (falls back to 1-VU as-fast-as-possible loop;
   observed 409k requests + threshold abort). Use `K6_TEST_DURATION` env var.
2. **Thresholds abort the run by default** (`abortOnFail`); use expected-
   response filters and no abort thresholds for continuity.
3. **Port-forward is the flaky link** on a contended host (TLS handshake
   timeouts kill `kubectl port-forward`). The in-cluster channel is the
   attribution path; k6+pf is for steady-state baselines only.

## Reconciliation (2026-09-15, issue #15) — artifact-driven ledger

Every current k6 run ships artifacts in `out/<run>/`: `raw.jsonl` (k6
`--out json`), `windows.json` (patch/rollout epochs), `summary.txt`.
Phase stats are recomputed from `raw.jsonl` + `windows.json`
(`split_phases.py compute()`), so any ledger row that reads "reconciled" was
re-derived from the points, not copied from a summary.

## Remaining for #1

- [x] migration-window k6 run with per-phase stats + in-cluster attribution
- [x] root-cause the claimed endpoint blackhole -> kubectl port-forward lifecycle
- [x] in-cluster k6 Job generator (`run_incluster.py`) as the standard continuity
      harness, streaming raw.jsonl via pod stdout (2026-09-15)
- [x] 50 rps stress pass via `run_incluster.py --duration 30s --rps 50`
      (2026-09-16): n=1453, 100%, p95 13.8 ms on the loaded host