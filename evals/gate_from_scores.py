"""Offline gate analysis over scored predictions exported by the Kaggle kernel.

For every model: pooled vs worst-group calibration of the unsafe gate on `cal`,
then risk among allowed / allow rate / unsafe-applied on test, ood, frozen.
Pure numpy/scipy; no model inference.

Usage: uv run python evals/gate_from_scores.py --scores evals/results/e1-kaggle.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from network_orch.cascade.gate import calibrate, calibrate_worst_group

RES = Path(__file__).resolve().parent / "results"
EVAL = ("test", "ood", "frozen")


def rows_by_split() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for line in (RES / "decision-dataset.jsonl").read_text().splitlines():
        r = json.loads(line)
        out.setdefault(r["split"], []).append(r)
    return out


def analyse(scores: dict, alphas=(0.10, 0.05, 0.01)) -> dict:
    data = rows_by_split()
    res: dict = {}
    for model, tasks in scores.items():
        pu = {s: [p[1] for p in tasks["unsafe"][s]] for s in ("cal",) + EVAL}
        y = {s: [r["unsafe"] for r in data[s]] for s in ("cal",) + EVAL}
        groups = [r["tmpl"] for r in data["cal"]]
        res[model] = {}
        for a in alphas:
            for name, cert in (
                ("pooled", calibrate(pu["cal"], y["cal"], a)),
                ("worst_group", calibrate_worst_group(pu["cal"], y["cal"], groups, a)),
            ):
                cell: dict = {"certified": cert.certified}
                if cert.certified:
                    cell["tau"] = round(cert.tau, 3)
                    for s in EVAL:
                        al = [u for p, u in zip(pu[s], y[s]) if p < cert.tau]
                        cell[s] = {"allow_rate": round(len(al) / len(y[s]), 3),
                                   "unsafe_applied": sum(al),
                                   "risk": round(sum(al) / len(al), 4) if al else 0.0}
                res[model][f"alpha={a}/{name}"] = cell
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(RES / "e1-kaggle.json"))
    ap.add_argument("--out", default=str(RES / "gate-analysis.json"))
    a = ap.parse_args()
    scores = json.loads(Path(a.scores).read_text())["scores"]
    out = analyse(scores)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for model, cells in out.items():
        for k, c in cells.items():
            if c["certified"]:
                print(model, k, "tau", c["tau"],
                      {s: (c[s]["risk"], c[s]["allow_rate"]) for s in EVAL})
            else:
                print(model, k, "UNCERTIFIED")


if __name__ == "__main__":
    main()
