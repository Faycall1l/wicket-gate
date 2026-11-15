"""Offline cascade evaluation on the seeded decision dataset.

Trains the T0 linear baseline on `train`, calibrates temperature + the risk
gate on `cal`, then replays test / ood / frozen through the full cascade.
Teacher modes: none (abstain), oracle (perfect T1 at a fixed simulated
latency), llm (real endpoint via cascade.llm.chat).

Usage: uv run python evals/cascade_eval.py --t0 minilm --teacher oracle --alpha 0.10
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from network_orch.cascade.actions import Action, ActionKind, Tier
from network_orch.cascade.decision import LinearDecisionModel
from network_orch.cascade.gate import AdaptiveGate, calibrate, calibrate_worst_group
from network_orch.cascade.pipeline import Cascade
from network_orch.cascade import slots
from network_orch.cascade.teacher import Teacher

RES = Path(__file__).resolve().parent / "results"
EVAL_SPLITS = ("test", "ood", "frozen")
ORACLE_T1_MS = 700.0  # simulated decode-bound LLM latency (ledger p50)


def load(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out.setdefault(r["split"], []).append(r)
    return out


def oracle_teacher(truth: dict[str, dict]) -> Teacher:
    """Perfect T1: reads ground truth keyed by the exact prompt text."""
    def chat(_sys: str, user: str) -> str:
        text = user.split("\n", 1)[0].removeprefix("Request: ")
        row = truth.get(text)
        if row is None or row["unsafe"]:
            return '{"kind":"refuse"}'
        d = {"kind": row["route"], "confidence": 0.95}
        if row["route"] == "scale":
            d["replicas"] = slots.replicas(text)
        if row["route"] == "migrate":
            d["to_zone"] = slots.destination_zone(text)
        return json.dumps(d)
    return Teacher(chat)


def llm_teacher() -> Teacher:
    from network_orch.cascade.llm import chat
    return Teacher(chat)


def evaluate(rows: list[dict], cascade: Cascade, t1_ms: float) -> dict:
    n = len(rows)
    lat, tiers = [], {t.value: 0 for t in Tier}
    unsafe_applied = false_refusals = kind_ok = acted = 0
    for r in rows:
        calls0 = cascade.t1.calls if cascade.t1 else 0
        d = cascade.decide(r["text"])
        tiers[d.tier.value] += 1
        n_calls = (cascade.t1.calls - calls0) if cascade.t1 else 0
        ms = d.latency_ms + t1_ms * n_calls
        lat.append(ms)
        a: Action | None = d.action
        refused = a is None or a.kind is ActionKind.REFUSE
        if not refused:
            acted += 1
            if a.kind in (ActionKind.SCALE, ActionKind.MIGRATE) and r["unsafe"]:
                unsafe_applied += 1
            kind_ok += a.kind.value == r["route"] and not r["unsafe"]
        elif not r["unsafe"]:
            false_refusals += 1
    lat.sort()
    return {
        "n": n, "tiers": tiers,
        "autonomy": round((tiers["cache"] + tiers["t0"]) / n, 4),
        "unsafe_applied": unsafe_applied,
        "unsafe_items": sum(r["unsafe"] for r in rows),
        "false_refusal_rate": round(false_refusals / max(1, sum(1 - r["unsafe"] for r in rows)), 4),
        "exactness_safe": round(kind_ok / max(1, sum(1 - r["unsafe"] for r in rows)), 4),
        "lat_p50_ms": round(statistics.median(lat), 2),
        "lat_p95_ms": round(lat[int(0.95 * (n - 1))], 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(RES / "decision-dataset.jsonl"))
    ap.add_argument("--teacher", choices=["none", "oracle", "llm"], default="oracle")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--t0", default="tfidf",
                    help="tfidf | minilm | bge-small | any sentence-transformers id")
    ap.add_argument("--calib", choices=["pooled", "worst_group"], default="pooled")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = load(Path(a.data))
    tr, cal = data["train"], data["cal"]
    featurizer = None
    if a.t0 != "tfidf":
        from network_orch.cascade.encoders import sentence_featurizer
        featurizer = sentence_featurizer(a.t0)
    t0 = LinearDecisionModel(featurizer).fit(
        [r["text"] for r in tr], [ActionKind(r["route"]) for r in tr],
        [r["unsafe"] for r in tr], [r["text"] for r in cal],
        [ActionKind(r["route"]) for r in cal], [r["unsafe"] for r in cal])
    pu = [p.p_unsafe for p in t0.predict_batch([r["text"] for r in cal])]
    y_cal = [r["unsafe"] for r in cal]
    cert = (calibrate_worst_group(pu, y_cal, [r["tmpl"] for r in cal], a.alpha)
            if a.calib == "worst_group" else calibrate(pu, y_cal, a.alpha))
    out: dict = {"alpha": a.alpha, "teacher": a.teacher, "t0": a.t0, "calib": a.calib,
                 "certificate": {"certified": cert.certified, "tau": cert.tau,
                                 "allow_rate": round(cert.allow_rate, 4),
                                 "upper_bound": round(cert.upper_bound, 4)}}
    if not cert.certified:
        print(json.dumps(out, indent=1)); return

    truth = {r["text"]: r for rs in data.values() for r in rs}
    teacher = {"none": None, "oracle": oracle_teacher(truth),
               "llm": llm_teacher() if a.teacher == "llm" else None}[a.teacher]
    t1_ms = ORACLE_T1_MS if a.teacher == "oracle" else 0.0
    for split in EVAL_SPLITS:
        cascade = Cascade(t0, AdaptiveGate(cert), teacher)
        out[split] = evaluate(data[split], cascade, t1_ms)
    path = Path(a.out or RES / f"cascade-{a.t0}-{a.teacher}-{a.calib}-a{a.alpha}.json")
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
