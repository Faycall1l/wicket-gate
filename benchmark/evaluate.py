"""Score the strict validator and the graded verdicts against cluster-measured harm labels (offline).

usage: python benchmark/evaluate.py [set1-oracle.jsonl set2-oracle.jsonl]
Each item's structured ground-truth action is checked; `harm` is the outcome of applying it to the
oracle Deployment on a real cluster. Reports harmful actions passed and safe actions blocked.
"""
import json, sys
from pathlib import Path

from network_orch.cascade import shield, slots
from network_orch.cascade.actions import Action, ActionKind
from network_orch.cascade.graded import ALLOW, decide

HERE = Path(__file__).parent
ZONES = ("edge-a", "edge-b", "edge-c", "cloud")


def action(it):
    ns = it.get("namespace") or "streaming-demo"
    if it["kind"] == "scale":
        return Action(ActionKind.SCALE, namespace=ns, replicas=it.get("replicas"))
    if it["kind"] == "migrate":
        return Action(ActionKind.MIGRATE, namespace=ns, to_zone=it.get("to_zone"))
    return Action(ActionKind.MONITOR)


def main(files):
    for f in files:
        items = [i for i in map(json.loads, (HERE / f).read_text().splitlines()) if i.get("harm") is not None]
        harmful = sum(i["harm"] for i in items); safe = len(items) - harmful
        row = {"strict": [0, 0], "graded": [0, 0]}
        for it in items:
            act = action(it)
            zone = it.get("from_zone") if it.get("from_zone") in ZONES else "edge-a"
            blocked_s = act.kind is not ActionKind.MONITOR and not shield.check(act, zone).ok
            blocked_g = decide(act, slots.underspecified(it["text"]))[0] != ALLOW
            for name, b in (("strict", blocked_s), ("graded", blocked_g)):
                row[name][0] += (not b) and it["harm"] == 1   # harmful passed
                row[name][1] += b and it["harm"] == 0         # safe blocked
        print(f"{f}: n={len(items)} harmful={harmful} safe={safe}")
        for name, (hp, sb) in row.items():
            print(f"  {name:6s} harmful passed {hp}/{harmful}   safe blocked {sb}/{safe} ({sb/safe:.1%})")


if __name__ == "__main__":
    main(sys.argv[1:] or ["set1-oracle.jsonl", "set2-oracle.jsonl"])
