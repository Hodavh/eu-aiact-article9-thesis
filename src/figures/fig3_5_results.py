"""Draws the paper's Figures 3-5 (fusion comparison, degradation curve, ablation) from results/paper/.
Usage: python src/figures/fig3_5_results.py [results_dir] [output_dir]"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
R = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "paper"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY, INK, EDGE, RED = "#2b6cb0", "#dd6b20", "#718096", "#1a202c", "#4a5568", "#c53030"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.edgecolor": EDGE, "axes.labelcolor": INK,
                     "xtick.color": EDGE, "ytick.color": EDGE, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
def grid(ax):
    ax.yaxis.grid(True, color="#e2e8f0", lw=0.5); ax.set_axisbelow(True)

# Figure 3: fusion strategies under structured noise
st = json.load(open(R / "layer3_structured.json"))
conds = ["symmetric_30_30", "asym_20_10", "asym_50_10", "asym_70_10"]
labels = ["Symmetric\n0.30 / 0.30", "Asymmetric\n0.20 / 0.10", "Asymmetric\n0.50 / 0.10", "Asymmetric\n0.70 / 0.10"]
fig, ax = plt.subplots(figsize=(5.5, 2.4)); w = 0.25
for i, (m, col, lab) in enumerate([("static", GREY, "Static"), ("attention", ORANGE, "Attention"), ("noise_aware", BLUE, "Noise-aware")]):
    xs = [k + (i - 1) * w for k in range(len(conds))]
    ax.bar(xs, [st[c][m] for c in conds], w * 0.92, yerr=[st[c][m + "_std"] for c in conds], color=col, label=lab,
           error_kw=dict(lw=0.6, capsize=1.8, capthick=0.6, ecolor=INK))
ax.set_xticks(range(len(conds))); ax.set_xticklabels(labels)
ax.set_ylabel("Accuracy"); ax.set_ylim(0.5, 0.9); grid(ax)
ax.set_xlabel("Nominal feature / structural noise level", labelpad=4)
ax.legend(ncol=3, loc="upper right", fontsize=6.5, handlelength=1.2, columnspacing=1.2)
fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_fusion.png", dpi=600, facecolor="white"); plt.close(fig)

# Figure 4: degradation curve
dg = json.load(open(R / "layer3_degradation.json"))["curve"]
lv = sorted(dg, key=float)
fig, ax = plt.subplots(figsize=(5.0, 2.4))
ax.axhline(0.70, ls=(0, (4, 2)), lw=0.8, color=RED, label="Usable threshold (0.70)")
ax.errorbar([float(k) for k in lv], [dg[k]["mean"] for k in lv], yerr=[dg[k]["std"] for k in lv], color=BLUE, lw=1.0,
            marker="o", ms=3.2, capsize=2, elinewidth=0.6, capthick=0.6, label="Accuracy (mean ± SD, 5 folds)")
ax.set_xlabel("Noise level (feature = structural)"); ax.set_ylabel("Classification accuracy")
ax.set_ylim(0.5, 1.0); ax.set_xticks([float(k) for k in lv]); grid(ax)
ax.legend(loc="upper right", fontsize=6.5)
fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_degradation.png", dpi=600, facecolor="white"); plt.close(fig)

# Figure 5: ablation with the all-non-compliant reference
ab = json.load(open(R / "ablation.json"))
names = ["Layer 3 only", "Layer 2 + Layer 3", "Reference: all\nnon-compliant"]
vals = [ab["L3_only"], ab["L2+L3"], ab.get("all_noncompliant_reference", 92 / 120)]
fig, ax = plt.subplots(figsize=(3.3, 2.4))
bars = ax.bar(range(3), vals, 0.6, color=[GREY, BLUE, "white"], edgecolor=[GREY, BLUE, EDGE], linewidth=0.8)
bars[2].set_hatch("////"); bars[2].set_edgecolor(EDGE)
for i, v in enumerate(vals): ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=6.3)
ax.set_ylabel("Verdict accuracy (120 test packages)"); ax.set_ylim(0, 1.0); grid(ax)
fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_ablation.png", dpi=600, facecolor="white"); plt.close(fig)
print("Saved Figures 3-5 to", OUT)
