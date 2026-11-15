#!/usr/bin/env python3
"""Aggregate multi-seed LLM-arm runs into pooled metrics with Wilson CIs.

Reads evals/results/llm-arms-<model>-<t0>-s<seed>.json for seeds 7/11/13,
pools counts per arm x split, and writes a markdown table + JSON.

Usage: uv run python evals/arms_aggregate.py --model openai_gpt-oss-20b --t0 bge-small
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
SEEDS = (7, 11, 13)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model tag as in the filename")
    ap.add_argument("--t0", default="bge-small")
    a = ap.parse_args()

    files = []
    for s in SEEDS:
        p = RES / f"llm-arms-{a.model}-{a.t0}-s{s}.json"
        if p.exists():
            files.append(p)
        else:
            print("MISSING", p)
    arms: dict[str, dict[str, dict]] = {}
    for f in files:
        d = json.loads(f.read_text())
        for arm, splits in d["arms"].items():
            for split, m in splits.items():
                agg = arms.setdefault(arm, {}).setdefault(
                    split, {"n": 0, "safe": 0, "correct": 0, "refused": 0,
                            "unsafe_items": 0, "complied_unsafe": 0,
                            "calls": 0.0, "underspec_refused": 0})
                agg["n"] += m["n"]
                agg["safe"] += m["safe_items"]
                agg["correct"] += m["correct_safe"]
                agg["refused"] += m["refused_safe"]
                agg["unsafe_items"] += m["unsafe_items"]
                agg["complied_unsafe"] += m["complied_unsafe"]
                agg["calls"] += m["llm_calls_per_intent"] * m["n"]
                agg["underspec_refused"] += m.get("underspec_refused", 0)

    print(f"seeds pooled: {[f.name for f in files]}\n")
    hdr = "arm | split | n | exact [CI] | false-ref [CI] | complied unsafe | calls/intent | unspec"
    print(hdr)
    print("-" * len(hdr))
    for arm in sorted(arms):
        for split in ("test", "ood", "frozen"):
            m = arms[arm].get(split)
            if not m:
                continue
            lo, hi = wilson(m["correct"], m["safe"])
            flo, fhi = wilson(m["refused"], m["safe"])
            print(f"{arm} | {split} | {m['n']} | "
                  f"{m['correct']/m['safe']:.3f} [{lo:.3f},{hi:.3f}] | "
                  f"{m['refused']/m['safe']:.3f} [{flo:.3f},{fhi:.3f}] | "
                  f"{m['complied_unsafe']}/{m['unsafe_items']} | "
                  f"{m['calls']/m['n']:.2f} | {m['underspec_refused']}")


if __name__ == "__main__":
    main()
