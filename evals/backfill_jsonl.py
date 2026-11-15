"""Append checkpoint (arm, split) entries missing from results/llm-arms.jsonl."""
import json, pathlib
R = pathlib.Path(__file__).parent / "results"
J = R / "llm-arms.jsonl"
seen = set()
if J.exists():
    for l in J.read_text().splitlines():
        r = json.loads(l); seen.add((r["model"], r["seed"], r["arm"], r["split"]))
with J.open("a") as f:
    for p in sorted(R.glob("llm-arms-*-s*.json")):
        d = json.loads(p.read_text())
        if "seed" not in d:
            continue
        for arm, sps in d["arms"].items():
            for sp, summ in sps.items():
                if (d["model"], d["seed"], arm, sp) not in seen:
                    f.write(json.dumps({"model": d["model"], "seed": d["seed"], "t0": d["t0"],
                        "alpha": d["alpha"], "n": d["n_per_split"], "tau": d["tau"],
                        "arm": arm, "split": sp, "summary": summ}) + "\n")
