"""Where the 400 unseen intents (seed 31) end up with the graded checks, drawn from the arm-I records."""
import collections, json
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / "evals/results/llm-arms-items.jsonl"
c = collections.Counter()
for line in open(RES):
    r = json.loads(line)
    if r["split"] == "llmgen" and r["seed"] == 31 and r["arm"] == "I" and r["model"] == "openai_gpt-oss-20b":
        lane = {"ALLOW": "applied", "ASK": "asked", "ESCALATE": "escalated"}.get(r["verdict"], "denied")
        c[(lane, "harmful" if r["unsafe"] else "safe")] += 1
repaired = sum(1 for line in open(RES) if (r := json.loads(line)) and r["split"] == "llmgen" and r["seed"] == 31
               and r["arm"] == "I" and r["model"] == "openai_gpt-oss-20b" and r["verdict"] == "ALLOW" and r["retries"])
lanes = ["applied", "asked", "escalated", "denied"]
INK, MUTED, BG, HARM, SAFE = "#1b1f2a", "#6b7080", "#fbfaf7", "#b4532a", "#2f6f6a"
K, GAP, X0, X1, W = 1.45, 34, 330, 900, 22
tot = {k: sum(v for (l, kk), v in c.items() if kk == k) for k in ("harmful", "safe")}
top = 110
# left nodes
ly = {"harmful": top, "safe": top + tot["harmful"] * K + 70}
# right nodes
ry, y = {}, top
for l in lanes:
    h = sum(v for (ll, _), v in c.items() if ll == l) * K
    ry[l] = (y, max(h, 14)); y += max(h, 14) + GAP
H = int(max(y, ly["safe"] + tot["safe"] * K) + 120)
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1560 {H}" font-family="Helvetica, Arial, sans-serif" fill="{INK}">',
     f'<rect width="1560" height="{H}" fill="{BG}"/>',
     f'<text x="40" y="48" font-size="26" font-weight="700">Where 400 unseen intents end up</text>',
     f'<text x="40" y="76" font-size="16" fill="{MUTED}">graded checks in front of gpt-oss-20b, each applied action run on a real cluster</text>']
lo = {k: ly[k] for k in ly}; ro = {l: ry[l][0] for l in lanes}
for kind, col in (("harmful", HARM), ("safe", SAFE)):
    for l in lanes:
        v = c.get((l, kind), 0)
        if not v:
            continue
        h = max(v * K, 3)
        a, b = lo[kind], ro[l]
        o.append(f'<path d="M{X0+W} {a:.1f}C{(X0+W+X1)/2} {a:.1f} {(X0+W+X1)/2} {b:.1f} {X1} {b:.1f}V{b+h:.1f}'
                 f'C{(X0+W+X1)/2} {b+h:.1f} {(X0+W+X1)/2} {a+h:.1f} {X0+W} {a+h:.1f}Z" fill="{col}" fill-opacity="0.45"/>')
        lo[kind] += v * K; ro[l] += h
for kind, col, label in (("harmful", HARM, "harmful intents"), ("safe", SAFE, "safe intents")):
    o.append(f'<rect x="{X0}" y="{ly[kind]:.1f}" width="{W}" height="{tot[kind]*K:.1f}" fill="{col}"/>')
    o.append(f'<text x="{X0-16}" y="{ly[kind]+tot[kind]*K/2-4:.1f}" text-anchor="end" font-size="30" font-weight="700" fill="{col}">{tot[kind]}</text>')
    o.append(f'<text x="{X0-16}" y="{ly[kind]+tot[kind]*K/2+22:.1f}" text-anchor="end" font-size="16" fill="{MUTED}">{label}</text>')
desc = {"applied": ("applied", "0 of 39 harmful-intent actions harmed the cluster"),
        "asked": ("asked", "operator clarifies a missing or ambiguous parameter"),
        "escalated": ("escalated", "scale to zero waits for approval"),
        "denied": ("denied", "protected namespace or agent refusal")}
for l in lanes:
    y0, h = ry[l]; n = sum(v for (ll, _), v in c.items() if ll == l)
    o.append(f'<rect x="{X1}" y="{y0:.1f}" width="{W}" height="{h:.1f}" fill="{INK}"/>')
    ty = y0 + h / 2
    o.append(f'<text x="{X1+W+18}" y="{ty-2:.1f}" font-size="26" font-weight="700">{n} <tspan font-size="20" font-weight="400">{desc[l][0]}</tspan></text>')
    o.append(f'<text x="{X1+W+18}" y="{ty+22:.1f}" font-size="15" fill="{MUTED}">{desc[l][1]}</text>')
o.append(f'<text x="40" y="{H-30}" font-size="14" fill="{MUTED}">{repaired} proposals were applied only after a repair retry. Intents are LLM-written; harm is rejection, protected write or not ready in 60 s.</text>')
o.append("</svg>")
(Path(__file__).parent / "flow.svg").write_text("\n".join(o))
