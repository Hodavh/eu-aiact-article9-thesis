"""Build a balanced diffuse-violation dataset for RQ3."""
import json, shutil, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inject_diffuse import inject_diffuse

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data"/"synthetic"/"base"
OUT=ROOT/"data"/"synthetic"/"diffuse"
LABELS=ROOT/"data"/"synthetic"/"labels_diffuse.json"

def build(n=600, seed=42):
    random.seed(seed)
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    labels={}
    base_pkgs=[d.name for d in BASE.iterdir() if d.name.startswith("PKG_C")]
    for i in range(n):
        src=BASE/random.choice(base_pkgs)
        # sample severity to get ~balanced classes: low sev -> compliant, high -> not
        sev=random.choice([0.0,0.1,0.15,0.2,   0.35,0.45,0.6,0.75])  # half low, half high
        pid=f"PKG_D{i:04d}"
        info=inject_diffuse(src, OUT/pid, severity=sev)
        labels[pid]={"label":info["label"],"defect_ratio":info["defect_ratio"]}
    json.dump(labels, open(LABELS,"w"), indent=2)
    from collections import Counter
    print(f"Built {len(labels)} diffuse packages")
    print("Balance:", Counter(v["label"] for v in labels.values()))

if __name__=="__main__":
    build()
