"""Build the base synthetic dataset: compliant + non-compliant packages with labels."""
import json, shutil, random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_package import make_package
from inject_violation import VIOLATIONS, inject

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "synthetic" / "base"
LABELS = ROOT / "data" / "synthetic" / "labels.json"

def build(n_compliant=100, n_per_violation=20, seed=42):
    random.seed(seed)
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    labels = {}

    # compliant packages
    for i in range(n_compliant):
        pid = f"PKG_C{i:04d}"
        make_package(pid, OUT/pid, n_risks=random.randint(6,12))
        labels[pid] = {"label":"compliant", "violated_requirement":None, "violation_type":None}

    # non-compliant: n per violation type
    tmp = OUT/"_tmp_compliant"
    make_package("tmp", tmp, n_risks=8)
    idx = 0
    for vt in VIOLATIONS:
        for j in range(n_per_violation):
            pid = f"PKG_N{idx:04d}"
            # make a fresh compliant base each time for variety
            base = OUT/f"_base_{idx}"
            make_package("base", base, n_risks=random.randint(6,12))
            info = inject(base, OUT/pid, vt)
            shutil.rmtree(base)
            labels[pid] = {"label":"non_compliant",
                "violated_requirement":info["violated_requirement"], "violation_type":vt}
            idx += 1
    shutil.rmtree(tmp)

    json.dump(labels, open(LABELS,"w"), indent=2)
    n_total = len(labels)
    n_comp = sum(1 for v in labels.values() if v["label"]=="compliant")
    print(f"Built {n_total} packages: {n_comp} compliant, {n_total-n_comp} non-compliant")
    print(f"Labels saved: {LABELS}")

if __name__ == "__main__":
    build()
