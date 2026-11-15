"""Evaluate the validator and T0 on LLM-written intents labeled by the cluster oracle.

Ground truth per item: structured action (kind, replicas, to_zone, namespace) and a
harm label from applying it on a real cluster (results/llmgen-oracle.jsonl).
"""
import argparse, json, math
from collections import Counter
from pathlib import Path

from network_orch.cascade import shield
from network_orch.cascade.actions import Action, ActionKind
from network_orch.cascade.decision import LinearDecisionModel
from network_orch.cascade.gate import AdaptiveGate, calibrate
from network_orch.cascade.pipeline import Cascade

RES = Path(__file__).parent / "results"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def gt_action(it):
    if it["kind"] == "scale":
        return Action(ActionKind.SCALE, namespace=it.get("namespace") or "streaming-demo", replicas=it.get("replicas"))
    if it["kind"] == "migrate":
        return Action(ActionKind.MIGRATE, namespace=it.get("namespace") or "streaming-demo", to_zone=it.get("to_zone"))
    return Action(ActionKind.MONITOR)


class NoGate:
    def allow(self, p): return True
    def observe(self, u): pass


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--t0", default="bge-small")
    a = ap.parse_args()
    items = [json.loads(l) for l in (RES / "llmgen-oracle.jsonl").read_text().splitlines()]
    items = [i for i in items if i.get("harm") is not None]
    print("n labeled", len(items), "harm", sum(i["harm"] for i in items))

    # (a) validator on ground-truth actions vs the cluster oracle
    tp = fp = fn = tn = 0; miss = []
    for it in items:
        act = gt_action(it)
        zone = it.get("from_zone") if it.get("from_zone") in ("edge-a", "edge-b", "edge-c", "cloud") else "edge-a"
        v = shield.check(act, zone) if act.kind is not ActionKind.MONITOR else shield.Verdict(ok=True)
        blocked = not v.ok
        tp += blocked and it["harm"]; fp += blocked and not it["harm"]
        fn += (not blocked) and it["harm"]; tn += (not blocked) and not it["harm"]
        if (not blocked) and it["harm"]: miss.append((it["kind"], it.get("replicas"), it.get("to_zone"), it["text"][:80]))
    print(f"validator vs oracle: TP={tp} FP={fp} FN={fn} TN={tn} recall={tp/max(1,tp+fn):.3f}"
          f" CI{wilson(tp,tp+fn)} precision={tp/max(1,tp+fp):.3f}")
    for m in miss[:8]: print("  missed:", m)

    # (b) T0 + validator pipeline, no LLM
    rows = [json.loads(l) for l in (RES / "decision-dataset.jsonl").read_text().splitlines()]
    tr = [r for r in rows if r["split"] == "train"]; cal = [r for r in rows if r["split"] == "cal"]
    feat = None
    if a.t0 != "tfidf":
        from network_orch.cascade.encoders import sentence_featurizer
        feat = sentence_featurizer(a.t0)
    t0 = LinearDecisionModel(feat).fit([r["text"] for r in tr], [ActionKind(r["route"]) for r in tr],
        [r["unsafe"] for r in tr], [r["text"] for r in cal], [ActionKind(r["route"]) for r in cal],
        [r["unsafe"] for r in cal])
    cert = calibrate([p.p_unsafe for p in t0.predict_batch([r["text"] for r in cal])],
                     [r["unsafe"] for r in cal], 0.10)
    preds = t0.predict_batch([i["text"] for i in items])
    acc = sum(p.top.value == i["kind"] for p, i in zip(preds, items)) / len(items)
    print(f"T0[{a.t0}] kind accuracy on LLM-written intents: {acc:.3f} CI{wilson(round(acc*len(items)),len(items))}")
    for name, gate in (("gate", AdaptiveGate(cert)), ("no-gate", NoGate())):
        c = Cascade(t0, gate, None)
        out = Counter(); unsafe_applied = 0
        for it in items:
            d = c.decide(it["text"])
            g = gt_action(it)
            if d.action is None: out["deferred"] += 1; continue
            if d.action.kind in (ActionKind.REFUSE,): out["refused"] += 1; continue
            if d.action.kind is ActionKind.MONITOR:
                out["monitor"] += 1; continue
            match = (d.action.kind == g.kind and d.action.replicas == g.replicas
                     and d.action.to_zone == g.to_zone and d.action.namespace == g.namespace)
            if it["harm"] and match: out["unsafe_applied"] += 1; unsafe_applied += 1
            elif it["harm"]: out["applied_other_on_harmful_intent"] += 1
            elif match: out["applied_correct"] += 1
            else: out["applied_wrong_on_safe_intent"] += 1
        n = len(items); print(name, {k: v for k, v in out.items()},
              "| llm-call fraction if deferred->LLM:", round(out["deferred"] / n, 3),
              "| unsafe_applied CI", wilson(unsafe_applied, sum(i["harm"] for i in items)))


if __name__ == "__main__":
    main()
