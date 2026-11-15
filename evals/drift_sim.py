"""Drift simulation: static vs adaptive gate on a stream that shifts mid-way.

Stream = in-distribution test, then OOD, then frozen. The gate alone decides
(shield excluded) so the risk bound is isolated. Outcomes of *allowed* items
are fed back to the adaptive gate with probability `--feedback`.

Usage: uv run python evals/drift_sim.py --alpha 0.10 --feedback 1.0
"""

from __future__ import annotations

import argparse
import json
import random

from cascade_eval import RES, load

from network_orch.cascade.actions import ActionKind
from network_orch.cascade.decision import LinearDecisionModel
from network_orch.cascade.gate import AdaptiveGate, calibrate

PHASES = ("test", "ood", "frozen")


def run(alpha: float, feedback: float, gamma: float, seed: int) -> dict:
    data = load(RES / "decision-dataset.jsonl")
    tr, cal = data["train"], data["cal"]
    t0 = LinearDecisionModel().fit(
        [r["text"] for r in tr], [ActionKind(r["route"]) for r in tr],
        [r["unsafe"] for r in tr], [r["text"] for r in cal],
        [ActionKind(r["route"]) for r in cal], [r["unsafe"] for r in cal])
    pu = [p.p_unsafe for p in t0.predict_batch([r["text"] for r in cal])]
    cert = calibrate(pu, [r["unsafe"] for r in cal], alpha)
    if not cert.certified:
        return {"alpha": alpha, "certified": False}
    rng = random.Random(seed)
    out: dict = {"alpha": alpha, "feedback": feedback, "gamma": gamma,
                 "tau0": cert.tau, "phases": {}}
    for mode in ("static", "adaptive"):
        gate = AdaptiveGate(cert, gamma=gamma)
        res = {}
        for phase in PHASES:
            rows = data[phase][:]
            rng.shuffle(rows)
            preds = t0.predict_batch([r["text"] for r in rows])
            allowed = unsafe = 0
            for r, p in zip(rows, preds):
                if not gate.allow(p.p_unsafe):
                    continue
                allowed += 1
                unsafe += r["unsafe"]
                if mode == "adaptive" and rng.random() < feedback:
                    gate.observe(bool(r["unsafe"]))
            res[phase] = {"n": len(rows), "allowed": allowed, "unsafe_allowed": unsafe,
                          "risk": round(unsafe / allowed, 4) if allowed else 0.0,
                          "allow_rate": round(allowed / len(rows), 4),
                          "tau_end": round(gate.tau, 4)}
        out["phases"][mode] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--feedback", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = run(a.alpha, a.feedback, a.gamma, a.seed)
    p = RES / f"drift-a{a.alpha}-fb{a.feedback}.json"
    p.write_text(json.dumps(out, indent=1))
    for mode, ph in out.get("phases", {}).items():
        print(mode, {k: (v["risk"], v["allow_rate"], v["tau_end"]) for k, v in ph.items()})


if __name__ == "__main__":
    main()
