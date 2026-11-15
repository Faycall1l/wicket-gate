"""Regression guard: on the cluster-labeled benchmark no harmful ground-truth action may pass either check."""
import json
from pathlib import Path

from network_orch.cascade import shield, slots
from network_orch.cascade.actions import Action, ActionKind
from network_orch.cascade.graded import ALLOW, decide

BENCH = Path(__file__).parent.parent / "benchmark"
ZONES = ("edge-a", "edge-b", "edge-c", "cloud")


def action(it):
    ns = it.get("namespace") or "streaming-demo"
    if it["kind"] == "scale":
        return Action(ActionKind.SCALE, namespace=ns, replicas=it.get("replicas"))
    if it["kind"] == "migrate":
        return Action(ActionKind.MIGRATE, namespace=ns, to_zone=it.get("to_zone"))
    return Action(ActionKind.MONITOR)


def test_no_harmful_action_passes():
    n = 0
    for f in ("set1-oracle.jsonl", "set2-oracle.jsonl"):
        for it in map(json.loads, (BENCH / f).read_text().splitlines()):
            if it.get("harm") != 1:
                continue
            n += 1
            act = action(it)
            zone = it.get("from_zone") if it.get("from_zone") in ZONES else "edge-a"
            strict_pass = act.kind is ActionKind.MONITOR or shield.check(act, zone).ok
            graded_pass = decide(act, slots.underspecified(it["text"]))[0] == ALLOW
            assert not strict_pass, it["text"]
            assert not graded_pass, it["text"]
    assert n > 500
