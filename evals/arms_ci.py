"""Wilson 95% CIs for pooled-over-seeds rates from results/llm-arms.jsonl."""
import json, math, collections
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
agg = collections.defaultdict(lambda: collections.Counter())
for l in open("evals/results/llm-arms.jsonl"):
    r = json.loads(l); s = r["summary"]; g = agg[(r["model"], r["arm"], r["split"])]
    for k in ("safe_items", "correct_safe", "refused_safe", "unsafe_items", "complied_unsafe"): g[k] += s[k]
    g["seeds"] += 1
print("model arm split seeds | exact_safe [CI] | false_refusal [CI] | complied_unsafe [CI]")
for (m, a, sp), g in sorted(agg.items()):
    f = lambda k, n: f"{k}/{n}={k/n:.3f} [{wilson(k,n)[0]:.3f},{wilson(k,n)[1]:.3f}]" if n else "-"
    print(m[-18:], a, sp, g["seeds"], "|", f(g["correct_safe"], g["safe_items"]), "|",
          f(g["refused_safe"], g["safe_items"]), "|", f(g["complied_unsafe"], g["unsafe_items"]))
