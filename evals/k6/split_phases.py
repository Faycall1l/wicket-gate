#!/usr/bin/env python3
"""Split k6 --out json into steady/migration/steady-after phases by epoch.

Phases are bounded by windows.json (written by migration-window.sh):
  - steady:          t < patch_epoch
  - migration:       patch_epoch <= t < rollout_done_epoch
  - steady-after:    t >= rollout_done_epoch

Input lines: {"type":"Point","metric":"http_req_duration"|"http_req_failed",
              "data":{"time":ISO8601,"value":...}}
compute() returns per-phase {count, success, p50, p95, p99} rows; the CLI
prints them as a markdown table.
"""
import json
from datetime import datetime
from statistics import median, quantiles

def parse(t: str) -> float:
    s = t.replace("Z", "+00:00")
    if "." in s:
        head, _, tail = s.partition(".")
        frac, _, tz = tail.partition("+")
        frac = (frac + "000000")[:6]
        s = f"{head}.{frac}+{tz}"
    return datetime.fromisoformat(s).timestamp()

def compute(path: str, windows_path: str) -> dict:
    w = json.load(open(windows_path))
    p0, p1 = w["patch_epoch"], w["rollout_done_epoch"]
    lats = {"steady": [], "migration": [], "steady-after": []}
    fails = {k: 0 for k in lats}
    counts = {k: 0 for k in lats}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "Point":
                continue
            metric = obj.get("metric")
            if metric not in ("http_req_duration", "http_req_failed"):
                continue
            group = obj["data"]["tags"].get("name", "")
            if group != "http://localhost:18080/":
                continue
            ts = parse(obj["data"]["time"])
            phase = "steady" if ts < p0 else "migration" if ts < p1 else "steady-after"
            counts[phase] += 1
            v = obj["data"]["value"]
            if metric == "http_req_duration":
                lats[phase].append(v)
            else:
                fails[phase] += int(v)

    rows = {}
    for phase in lats:
        n = counts[phase]
        if n == 0:
            rows[phase] = {"count": 0, "success": None, "p50": None, "p95": None, "p99": None}
            continue
        lt = lats[phase]
        rows[phase] = {
            "count": n,
            "success": 1 - fails[phase] / n,
            "p50": median(lt),
            "p95": quantiles(lt, n=20)[18],
            "p99": quantiles(lt, n=100)[98],
        }
    return rows

def main(path: str, windows_path: str) -> int:
    rows = compute(path, windows_path)
    print("| phase | requests | success | p50 ms | p95 ms | p99 ms |")
    print("|---|---|---|---|---|---|")
    ok = True
    for phase, r in rows.items():
        if r["count"] == 0:
            print(f"| {phase} | 0 | - | - | - | - |")
            continue
        s = r["success"]
        if s < 1:
            ok = False
        print(f"| {phase} | {r['count']} | {s:.4f} | {r['p50']:.1f} | {r['p95']:.1f} | {r['p99']:.1f} |")
    return 0 if ok else 1

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1], sys.argv[2]))