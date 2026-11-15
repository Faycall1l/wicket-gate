#!/usr/bin/env python3
"""Run the k6 continuity script as an in-cluster Job (issue #1 standard
harness path) and pull back machine-readable artifacts.

Replaces the busybox seq-loader as the continuity generator: k6 runs inside the
cluster against svc/streamer, so it exercises the real Service endpoints (no
host<->cluster port-forward in the data path). The container streams raw.jsonl
to stdout (sh -c + cat), so the raw metric stream is harvested from the pod
logs even after the pod has completed.

Usage:
  uv run python evals/k6/run_incluster.py --duration 90s
  uv run python evals/k6/run_incluster.py --url http://stateful:8080/value
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

NS = "streaming-demo"
JOB = "k6-continuity"
CM = "k6-script"
IMG = "grafana/k6:0.53.0"
K6DIR = Path(__file__).resolve().parent
SCRIPT = K6DIR / "continuity.js"
OUT = K6DIR / "out"

MANIFEST = """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: {cm}
  namespace: {ns}
data:
  continuity.js: |
{js}
---
apiVersion: batch/v1
kind: Job
metadata:
  name: {job}
  namespace: {ns}
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: k6
        image: {img}
        command: ["sh", "-c"]
        args: ["k6 run --no-thresholds --summary-export=/tmp/summary.json --out json=/tmp/raw.jsonl /scripts/continuity.js; cat /tmp/raw.jsonl"]
        env:
        - name: K6_TARGET_URL
          value: "{url}"
        - name: K6_TEST_DURATION
          value: "{duration}"
        - name: K6_RATE
          value: "{rps}"
        resources:
          limits: {{cpu: 500m, memory: 256Mi}}
          requests: {{cpu: 100m, memory: 64Mi}}
        volumeMounts:
        - name: script
          mountPath: /scripts
      volumes:
      - name: script
        configMap:
          name: {cm}
"""


def kubectl(*args: str, timeout: int = 90, retries: int = 4) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = subprocess.run(["kubectl", "--request-timeout=60s", *args],
                               capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return r.stdout.strip()
            last = RuntimeError(r.stderr.strip()[:300])
        except subprocess.TimeoutExpired as e:
            last = e
        time.sleep(6 * (attempt + 1))
    raise RuntimeError(f"kubectl {' '.join(args[:6])} failed: {last}")


def kget(*args: str, timeout: int = 40) -> str:
    try:
        r = subprocess.run(["kubectl", "--request-timeout=30s", *args],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def kube_del(*args: str) -> None:
    """Best-effort delete: NotFound is fine (first run, already gone)."""
    try:
        kubectl("delete", *args, "--wait=false")
    except RuntimeError as e:
        if "NotFound" not in str(e):
            raise


def _balanced_end(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return -1


def _extract_summary(head: str) -> dict:
    best: dict = {}
    for i in [m.start() for m in re.finditer(r"\{", head)]:
        j = _balanced_end(head, i)
        if j < 0:
            continue
        try:
            obj = json.loads(head[i:j + 1])
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if "steady" in obj or ("requests" in obj and "success_rate" in obj):
            return obj
        if len(obj) > len(best):
            best = obj
    return best


def parse_logs(logs: str, ts: str, run_dir: Path, url: str, duration: str,
               rps: int | None = None, ns: str = NS) -> dict:
    """Harvest k6 summary + raw metric stream from the pod stdout into
    summary.json / scored.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "k6-stdout.txt").write_text(logs)

    # The container streams raw.jsonl to stdout after k6 ends, so the
    # handleSummary JSON is the brace-balanced object before the raw lines.
    head = logs.split('{"metric":', 1)[0]
    summary = _extract_summary(head)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    raw_points = [line for line in logs.splitlines()
                  if line.startswith('{"metric":') and '"Point"' in line]
    if not raw_points:
        raise RuntimeError("no k6 raw points found in logs (raw.jsonl was streamed "
                           "via 'cat /tmp/raw.jsonl' in the container)")
    counts = {"requests": 0, "ok": 0, "lats": []}
    for line in raw_points:
        o = json.loads(line)
        metric = o.get("metric")
        if metric == "http_req_duration":
            counts["requests"] += 1
            counts["lats"].append(o["data"]["value"])
        elif metric == "http_reqs" and o.get("data", {}).get("tags", {}).get("expected_response") == "true":
            counts["ok"] += 1
    lats = sorted(counts["lats"])
    q = lambda f: lats[min(int(f * len(lats)), len(lats) - 1)] if lats else None
    scored = {
        "run": ts, "mode": "in-cluster", "target": url, "namespace": ns,
        "duration": duration,
        "requests": counts["requests"],
        "success_rate": counts["ok"] / counts["requests"] if counts["requests"] else 0,
        "p50_ms": round(q(0.50), 2) if q(0.50) is not None else None,
        "p95_ms": round(q(0.95), 2) if q(0.95) is not None else None,
        "p99_ms": round(q(0.99), 2) if q(0.99) is not None else None,
        "summary_from_logs": summary,
    }
    if rps is not None:
        scored["rps"] = rps
    (run_dir / "scored.json").write_text(json.dumps(scored, indent=2))
    return scored


def _wait_until(fn, timeout_s: int, poll_s: float = 4.0, label: str = "") -> str:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        v = fn()
        if v:
            if label:
                print(f"{label}: {str(v)[:90]}")
            return v
        time.sleep(poll_s)
    raise RuntimeError(f"timed out waiting for {label or 'condition'} ({timeout_s}s)")


def _build_manifest(url: str, duration: str, rps: int, ns: str = NS) -> str:
    js = SCRIPT.read_text().splitlines()
    return MANIFEST.format(
        cm=CM, ns=ns, job=JOB, img=IMG, url=url, duration=duration, rps=rps,
        js="\n".join("      " + ln for ln in js),
    )


def run(duration: str, url: str, rps: int = 20, ns: str = NS) -> dict:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUT / f"incluster-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    kube_del("job", JOB, "-n", ns)
    kube_del("cm", CM, "-n", ns)
    manifest = _build_manifest(url, duration, rps, ns)
    p = subprocess.run(["kubectl", "apply", "-f", "-"], input=manifest,
                       capture_output=True, text=True, timeout=90)
    if p.returncode != 0:
        raise RuntimeError(f"apply failed: {p.stderr.strip()[:300]}")

    pod = _wait_until(
        lambda: kget("-n", ns, "get", "pods", "-l", f"job-name={JOB}",
                     "-o", "jsonpath={.items[0].metadata.name}"),
        timeout_s=180, label="k6 pod")

    dur_s = _parse_duration(duration)
    _wait_until(
        lambda: kget("-n", ns, "get", "pods", "-l", f"job-name={JOB}",
                     "-o", "jsonpath={.items[0].status.phase}") == "Running"
        or kget("-n", ns, "get", "pods", "-l", f"job-name={JOB}",
                "-o", "jsonpath={.items[0].status.phase}") in ("Succeeded", "Failed"),
        timeout_s=300, poll_s=5, label="pod running")

    # wait for the Job to reach success criteria (k6 done + cat streamed)
    _wait_until(
        lambda: kget("-n", ns, "get", "job", JOB,
                     "-o", "jsonpath={.status.succeeded}") == "1",
        timeout_s=dur_s + 300, poll_s=5, label="job succeeded")

    logs = kubectl("-n", ns, "logs", pod, timeout=120)
    return parse_logs(logs, ts, run_dir, url, duration, rps, ns)


def _parse_duration(s: str) -> int:
    m = re.fullmatch(r"(\d+)(ms|s|m|h)", s)
    if not m:
        return 120
    v, unit = int(m.group(1)), m.group(2)
    return {"ms": 0, "s": v, "m": v * 60, "h": v * 3600}[unit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", default="90s")
    ap.add_argument("--url", default="http://streamer/")
    ap.add_argument("--rps", type=int, default=20)
    ap.add_argument("--ns", default=NS)
    args = ap.parse_args()
    out = run(args.duration, args.url, args.rps, args.ns)
    print(json.dumps(out, indent=2))
    if not out.get("requests"):
        sys.exit(1)
    run_id = out["run"]
    print(f"artifacts: {OUT / f'incluster-{run_id}'}")


if __name__ == "__main__":
    main()