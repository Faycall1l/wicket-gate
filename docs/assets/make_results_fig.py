"""Results figure from the cluster-measured numbers (fresh set, seed 31)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

arms = ["Agent alone", "+ strict validator", "+ graded checks"]
harmed = [120 / 196 * 100, 0, 0]
blocked = [7 / 204 * 100, 45 / 204 * 100, 8 / 204 * 100]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=200)
for a, vals, title, col in ((ax[0], harmed, "Harmful intents that reach harm (%)", "#dc2626"),
                            (ax[1], blocked, "Safe intents refused or escalated (%)", "#4f46e5")):
    bars = a.barh(arms[::-1], vals[::-1], color=col, height=0.55)
    a.set_title(title, fontsize=11, loc="left", fontweight="bold")
    a.set_xlim(0, 70 if col == "#dc2626" else 30)
    for b, v in zip(bars, vals[::-1]):
        a.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=10)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.tick_params(left=False)
ax[1].set_yticklabels([])
fig.text(0.01, -0.04, "400 LLM-written intents (196 harmful, 204 safe), gpt-oss-20b, harm measured on a k3d cluster.", fontsize=8, color="#555")
fig.tight_layout(); fig.savefig("docs/assets/results.png", bbox_inches="tight")
