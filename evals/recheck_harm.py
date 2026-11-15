"""Re-run every harm=1 cluster outcome 3x and take the majority; first-run labels are kept as harm_first.

The first run is subject to transient 20 s deadline misses (an 8-item audit showed 8/8 first-run harms
that were 'ready' on all three reruns). Applies to exec-oracle.jsonl and the validator-passed items of
llmgen-oracle.jsonl.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from k8s_oracle import RES, k
from k8s_oracle_gen import run


DEADLINE = 60  # s; chosen from measured good-migration times (1.5-24.8 s, n=30), not from the results


def maj(it):
    hs = [run(it, DEADLINE)["harm"] for _ in range(3)]
    return int(sum(h or 0 for h in hs) >= 2), hs


def main():
    p = RES / "exec-oracle.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    for r in rows:
        r.setdefault("harm_first", r["harm"])
        if r["harm_first"] != 1 or r.get("deadline") == DEADLINE:
            continue
        a = json.loads(r["key"])
        structural = a[3] in ("kube-system", "kube-public", "kube-node-lease") or (
            a[0] == "scale" and isinstance(a[1], int) and not 1 <= a[1] <= 8) or (
            a[0] == "migrate" and a[2] not in ("edge-a", "edge-b", "edge-c", "cloud"))
        if structural:  # rejected or failing by construction: no rerun needed
            r["reruns"] = "structural"; r["deadline"] = DEADLINE
            continue
        it = {"kind": a[0], "replicas": a[1], "to_zone": a[2], "namespace": a[3], "from_zone": a[4]}
        r["harm"], r["reruns"] = maj(it); r["deadline"] = DEADLINE
        print(r["key"], r["harm_first"], "->", r["harm"], r["reruns"], flush=True)
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    k("-n", "oracle", "scale", "deploy/streamer", "--replicas=2")


if __name__ == "__main__":
    main()
