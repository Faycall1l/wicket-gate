"""Paired exact McNemar tests between arms from results/llm-arms-items.jsonl.

Items are matched on (model, seed, split, item). Metrics: correct (safe items),
refused (safe items; false refusal), complied_unsafe (unsafe items).
Usage: python evals/arms_paired.py [ARM_X ARM_Y ...]  (default: E vs B, E vs D, E vs F/G/H)
"""
import collections
import json
import sys
from math import comb

PAIRS = [("E", "B"), ("E", "A"), ("E", "D"), ("E", "F"), ("E", "G"), ("E", "H")]
METRICS = (("correct", 0), ("refused", 0), ("complied_unsafe", 1))  # (metric, unsafe-subset?)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    rows = collections.defaultdict(dict)
    for line in open("evals/results/llm-arms-items.jsonl"):
        r = json.loads(line)
        rows[(r["model"], r["arm"])][(r["seed"], r["split"], r["item"])] = r
    pairs = PAIRS
    if len(sys.argv) > 2:
        pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
    print("model split X>Y: metric | X-only Y-only (discordant) | exact p")
    for model in sorted({m for m, _ in rows}):
        for x, y in pairs:
            X, Y = rows.get((model, x)), rows.get((model, y))
            if not X or not Y:
                continue
            for split in ("test", "ood", "frozen", "llmgen"):
                for metric, unsafe_subset in METRICS:
                    b = c = 0
                    for k in X.keys() & Y.keys():
                        if k[1] != split or bool(X[k]["unsafe"]) != bool(unsafe_subset):
                            continue
                        b += X[k][metric] > Y[k][metric]
                        c += X[k][metric] < Y[k][metric]
                    if b + c:
                        print(f"{model[-16:]:16} {split:6} {x}v{y} {metric:16} "
                              f"{b:3} {c:3}  p={mcnemar_exact(b, c):.4f}")


if __name__ == "__main__":
    main()
