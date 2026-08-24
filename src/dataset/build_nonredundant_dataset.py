"""Non-redundant dataset: violations split across channels.
Structural violation (R6): a mitigation link removed -> visible ONLY in edges.
Feature violation (R9): acceptability set not_acceptable -> visible ONLY in features.
Compliant = neither. Non-compliant = one or the other (50/50 split of which)."""
import json, csv, shutil, random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data"/"synthetic"/"base"
OUT=ROOT/"data"/"synthetic"/"nonredundant"
LABELS=ROOT/"data"/"synthetic"/"labels_nonredundant.json"

def _rc(p): return list(csv.DictReader(open(p)))
def _wc(p,rows):
    with open(p,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def make_structural_violation(d):
    """Remove a mitigation entirely -> a risk becomes unmitigated (edge-only signal)."""
    mp=d/"mitigation_tracker.csv"; rows=_rc(mp)
    if len(rows)>1: rows.pop(); _wc(mp,rows)

def make_feature_violation(d):
    """Set one acceptability to not_acceptable (feature-only signal)."""
    mp=d/"mitigation_tracker.csv"; rows=_rc(mp)
    rows[0]["acceptability"]="not_acceptable"; _wc(mp,rows)

def build(n=600, seed=42):
    random.seed(seed)
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    base_pkgs=[p.name for p in BASE.iterdir() if p.name.startswith("PKG_C")]
    labels={}
    for i in range(n):
        src=BASE/random.choice(base_pkgs)
        pid=f"PKG_NR{i:04d}"; dst=OUT/pid
        shutil.copytree(src,dst)
        roll=random.random()
        if roll<0.5:
            labels[pid]={"label":"compliant"}
        elif roll<0.75:
            make_structural_violation(dst); labels[pid]={"label":"non_compliant","kind":"structural"}
        else:
            make_feature_violation(dst); labels[pid]={"label":"non_compliant","kind":"feature"}
    json.dump(labels,open(LABELS,"w"),indent=2)
    from collections import Counter
    print(f"Built {len(labels)} packages")
    print("Labels:", Counter(v["label"] for v in labels.values()))
    print("Violation kinds:", Counter(v.get("kind") for v in labels.values() if v.get("kind")))

if __name__=="__main__": build()
