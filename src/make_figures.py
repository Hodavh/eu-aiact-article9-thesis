"""Generate publication figures from saved results JSON."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "results" / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})

# --- Figure 1: Degradation curve (RQ4) ---
try:
    deg = json.load(open(R / "layer3_degradation.json"))
    curve = deg["curve"]
    levels = sorted(float(k) for k in curve)
    means = [curve[str(l) if str(l) in curve else f"{l:.1f}"]["mean"]
             if (str(l) in curve or f"{l:.1f}" in curve) else None for l in levels]
    # robust key lookup
    means, stds = [], []
    for l in levels:
        key = next(k for k in curve if float(k) == l)
        means.append(curve[key]["mean"]); stds.append(curve[key]["std"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(levels, means, yerr=stds, marker="o", capsize=3, color="#2b6cb0", label="Accuracy")
    ax.axhline(0.70, ls="--", color="#c53030", label="Usable threshold (0.70)")
    ax.set_xlabel("Noise level (feature = structure)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Robustness to evidence degradation (RQ4)")
    ax.set_ylim(0.5, 1.0); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_degradation.png"); plt.close()
    print("Saved fig_degradation.png")
except Exception as e:
    print("Degradation figure skipped:", e)

# --- Figure 2: Fusion comparison across asymmetry (RQ3) ---
try:
    st = json.load(open(R / "layer3_structured.json"))
    conds = list(st.keys())
    labels = {"symmetric_30_30":"Sym\n30/30","asym_20_10":"Asym\n20/10",
              "asym_50_10":"Asym\n50/10","asym_70_10":"Asym\n70/10"}
    x = range(len(conds)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (m, col) in enumerate([("static","#718096"),("attention","#dd6b20"),("noise_aware","#2b6cb0")]):
        vals = [st[c][m] for c in conds]; errs = [st[c].get(f"{m}_std",0) for c in conds]
        ax.bar([xi + (i-1)*w for xi in x], vals, w, yerr=errs, capsize=2,
               label=m.replace("_"," "), color=col)
    ax.set_xticks(list(x)); ax.set_xticklabels([labels.get(c,c) for c in conds])
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.5, 0.9)
    ax.set_title("Fusion strategy under structured noise (RQ3)")
    ax.legend(title="Fusion")
    fig.tight_layout(); fig.savefig(OUT / "fig_fusion.png"); plt.close()
    print("Saved fig_fusion.png")
except Exception as e:
    print("Fusion figure skipped:", e)

# --- Figure 3: Ablation (RQ5) ---
try:
    ab = json.load(open(R / "ablation.json"))
    cfgs = ["L3_only","L2+L3"]; vals = [ab["L3_only"], ab["L2+L3"]]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(["Layer 3\nonly","Full\npipeline"], vals, color=["#718096","#2b6cb0"], width=0.6)
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center")
    ax.set_ylabel("Verdict accuracy"); ax.set_ylim(0.6, 0.85)
    ax.set_title("Ablation: contribution of layers (RQ5)")
    fig.tight_layout(); fig.savefig(OUT / "fig_ablation.png"); plt.close()
    print("Saved fig_ablation.png")
except Exception as e:
    print("Ablation figure skipped:", e)

print(f"\nFigures written to {OUT}")
