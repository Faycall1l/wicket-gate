"""System structure figure on an 8 px grid. Run to regenerate structure.svg.

Every text call can pass maxw; the script prints a warning when the estimated width exceeds it.
"""
from pathlib import Path

INK, MUTED, BG, CARD, EDGE, SHADOW = "#1b1f2a", "#6b7080", "#f7f5ef", "#ffffff", "#d6d0c2", "#e9e4d8"
TEAL, RUST, GOLD, SLATE = "#2f6f6a", "#b4532a", "#b07d1c", "#5b6275"
FONT, MONO = "Helvetica, Arial, sans-serif", "Menlo, Consolas, monospace"
o, warn = [], []


def est(s, size, weight=400, mono=False):
    return len(s) * size * (0.60 if mono else (0.56 if weight >= 600 else 0.52))


def text(x, y, s, size=14, weight=400, fill=INK, anchor="start", mono=False, maxw=None):
    w = est(s, size, weight, mono)
    if maxw and w > maxw:
        warn.append(f"overflow {w:.0f}>{maxw}: {s}")
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
             f'font-family="{MONO if mono else FONT}">{s}</text>')


def card(x, y, w, h, accent=None, r=16):
    o.append(f'<rect x="{x+3}" y="{y+4}" width="{w}" height="{h}" rx="{r}" fill="{SHADOW}"/>')
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{CARD}" stroke="{EDGE}" stroke-width="1.5"/>')
    if accent:
        o.append(f'<rect x="{x+14}" y="{y}" width="36" height="4" rx="2" fill="{accent}"/>')


def tab(x, y, w, label, col, h=30):
    """Verdict label attached to a card edge."""
    o.append(f'<rect x="{x}" y="{y-h/2}" width="{w}" height="{h}" rx="{h/2}" fill="{col}"/>')
    text(x + w / 2, y + 4.5, label, 13, 700, "#fff", "middle", maxw=w - 16)


def line(d, col, w=3, dash=None, head=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#h{head})"' if head else ""
    o.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{w}" stroke-linecap="butt" stroke-linejoin="round"{da}{mk}/>')


W, H = 1600, 790
o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"><defs>')
for k, c in (("ink", INK), ("teal", TEAL), ("rust", RUST), ("gold", GOLD), ("slate", SLATE)):
    o.append(f'<marker id="h{k}" viewBox="0 0 12 12" refX="11" refY="6" markerUnits="userSpaceOnUse" markerWidth="15" markerHeight="15" '
             f'orient="auto"><path d="M0 0L12 6L0 12z" fill="{c}"/></marker>')
o.append("</defs>")
o.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
text(40, 56, "System structure", 28, 700)
text(40, 84, "The checks sit between the agent and the cluster. The cluster keeps its own controls, and an oracle measures what got through.",
     16, 400, MUTED, maxw=1500)

# ---- MCP boundary (drawn first, below cards)
o.append(f'<rect x="312" y="176" width="710" height="468" rx="22" fill="none" stroke="{MUTED}" stroke-width="1.6" stroke-dasharray="8 7"/>')
text(1006, 632, "network_orch_mcp", 12.5, 600, MUTED, "end", True)

# ---- agent
card(40, 200, 230, 200, TEAL)
text(64, 242, "LLM agent", 21, 700, maxw=180)
text(64, 268, "reads the operator's text", 14, 400, MUTED, maxw=190)
text(64, 288, "and proposes a typed action", 14, 400, MUTED, maxw=190)
for i, s in enumerate(("scale", "migrate", "monitor")):
    x = 64 + i * 68
    o.append(f'<rect x="{x}" y="316" width="62" height="26" rx="13" fill="{BG}" stroke="{EDGE}"/>')
    text(x + 31, 333, s, 11.5, 600, INK, "middle", True, maxw=56)
text(64, 378, "no tool access of its own", 13, 400, MUTED, maxw=190)
line("M270 300H328", INK, 3.5, head="ink")

# ---- graded checks
card(330, 200, 310, 410, TEAL)
text(354, 242, "Graded checks", 21, 700, maxw=250)
text(354, 268, "deterministic code, no model in the loop", 14, 400, MUTED, maxw=270)
rules = [("replicas within the policy range", SLATE), ("destination zone exists", SLATE),
         ("parameters present and unambiguous", GOLD), ("scale to zero", GOLD), ("namespace not protected", RUST)]
y = 312
for r, c in rules:
    o.append(f'<circle cx="362" cy="{y-5}" r="5" fill="{c}"/>')
    text(378, y, r, 13.5, 500, maxw=240)
    y += 36
o.append(f'<path d="M354 {y-14}H616" stroke="{EDGE}"/>')
text(354, y + 12, "Policy", 14, 700)
text(354, y + 34, "min / max replicas, zones, protected", 13, 400, MUTED, maxw=262)
text(354, y + 52, "namespaces, optional allowlist", 13, 400, MUTED, maxw=262)
text(354, y + 84, "Repair loop: at most 2 retries", 13, 600, SLATE, maxw=262)

# ---- verdict tabs and routes
tab(624, 282, 92, "ALLOW", TEAL)
tab(624, 372, 152, "ASK · ESCALATE", GOLD)
line("M776 372H846", GOLD, 4, head="gold")
tab(624, 482, 76, "DENY", RUST)
line("M700 482H756", RUST, 4, head="rust")

card(850, 320, 152, 104, GOLD)
o.append(f'<circle cx="884" cy="360" r="11" fill="none" stroke="{INK}" stroke-width="2.2"/>')
o.append(f'<path d="M864 398c0-13 8-21 20-21s20 8 20 21z" fill="none" stroke="{INK}" stroke-width="2.2"/>')
text(912, 366, "Operator", 16, 700, maxw=84)
text(912, 388, "clarifies or", 13, 400, MUTED, maxw=84)
text(912, 406, "approves", 13, 400, MUTED, maxw=84)

card(760, 450, 242, 66, RUST)
text(782, 484, "Refused", 16, 700, maxw=200)
text(782, 504, "protected target or agent refusal", 12.5, 400, MUTED, maxw=206)

card(680, 548, 322, 62, SLATE)
text(702, 574, "Audit log", 15, 700, maxw=200)
text(702, 594, "every apply call, with its verdict", 12.5, 400, MUTED, maxw=282)

# ---- repair loop
line("M485 200V152H155V196", SLATE, 2.5, "7 6", "slate")
o.append(f'<rect x="250" y="136" width="232" height="32" rx="16" fill="{SLATE}"/>')
text(366, 157, "REPAIR: the reason goes back", 13, 700, "#fff", "middle", maxw=212)

# ---- kubernetes
card(1040, 200, 520, 366, TEAL, 18)
text(1066, 244, "Kubernetes (k3d)", 21, 700, maxw=260)
text(1536, 244, "native controls stay on", 13, 400, MUTED, "end", maxw=200)
bx = (1066, 1234, 1402)
for x, (lab, sub) in zip(bx, (("RBAC", "restricted account"), ("Admission", "two policies"), ("Deployment", "streamer"))):
    o.append(f'<rect x="{x}" y="270" width="136" height="60" rx="12" fill="{BG}" stroke="{EDGE}" stroke-width="1.5"/>')
    text(x + 68, 296, lab, 16, 700, INK, "middle", maxw=120)
    text(x + 68, 316, sub, 12, 400, MUTED, "middle", maxw=124)
line("M1206 300H1230", INK, 2.5, head="ink"); line("M1374 300H1398", INK, 2.5, head="ink")
# bus from Deployment to the four zones
tx = (1064, 1182, 1300, 1418)
cx = [x + 52 for x in tx]
line(f"M{bx[2]+68} 330V366", INK, 2.5)
line(f"M{cx[0]} 366H{cx[3]}", INK, 2.5)
for c in cx:
    line(f"M{c} 366V388", INK, 2.5, head="ink")
for i, (x, (z, lvl, dots)) in enumerate(zip(tx, (("edge-a", 66, 2), ("edge-b", 34, 0), ("edge-c", 34, 0), ("cloud", 34, 0)))):
    o.append(f'<rect x="{x}" y="392" width="104" height="124" rx="12" fill="{CARD}" stroke="{EDGE}" stroke-width="1.5"/>')
    o.append(f'<rect x="{x+3}" y="{513-lvl}" width="98" height="{lvl}" rx="9" fill="{TEAL}" fill-opacity="0.22"/>')
    text(x + 52, 416, z, 13.5, 700, INK, "middle", maxw=90)
    for d in range(dots):
        o.append(f'<circle cx="{x+38+d*28}" cy="486" r="8" fill="{TEAL}"/>')
text(1066, 546, "four zones; a migration reschedules pods by node label", 12.5, 400, MUTED, maxw=440)

# ALLOW route drawn after the cluster card so its head stays visible
line("M716 282H1062", TEAL, 5, head="teal")

# ---- oracle
card(1040, 612, 520, 128, RUST, 18)
o.append(f'<path d="M1072 690a28 28 0 0 1 56 0" fill="none" stroke="{INK}" stroke-width="3"/>')
o.append(f'<path d="M1100 690L1116 670" stroke="{RUST}" stroke-width="4" stroke-linecap="round"/><circle cx="1100" cy="690" r="5" fill="{INK}"/>')
text(1156, 646, "Cluster oracle", 18, 700, maxw=200)
text(1536, 646, "evaluation only", 12.5, 600, RUST, "end", True, maxw=150)
text(1156, 670, "applies each executed action to a real Deployment;", 13, 400, MUTED, maxw=380)
text(1156, 690, "harm is an API or RBAC rejection, a protected-namespace", 13, 400, MUTED, maxw=380)
text(1156, 710, "write, or not ready in 60 s (in range: 3-run majority)", 13, 400, MUTED, maxw=380)
line("M1300 570V606", RUST, 2.5, "1 7", "rust")

# ---- benchmark
card(330, 676, 692, 64, SLATE)
text(354, 702, "Benchmark", 15, 700)
text(354, 723, "1,242 LLM-written intents with cluster-measured harm, scored offline by benchmark/evaluate.py", 12.5, 400, MUTED, maxw=650)

o.append("</svg>")
Path(__file__).with_name("structure.svg").write_text("\n".join(o))
print("\n".join(warn) or "no overflow warnings")
