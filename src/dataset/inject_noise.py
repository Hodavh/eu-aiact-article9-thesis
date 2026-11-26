"""Inject graduated feature and structural noise into a package copy."""
import csv, json, shutil, random
from pathlib import Path

CSV_FILES = ["clinical_risk_register.csv","mitigation_tracker.csv",
    "clinical_validation_study.csv","post_market_surveillance.csv","traceability_matrix.csv"]

def _read_csv(p): return list(csv.DictReader(open(p)))
def _write_csv(p,rows):
    if not rows: return
    with open(p,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def feature_noise(d, level):
    """Blank out a `level` fraction of field values across CSV files."""
    for fn in CSV_FILES:
        p = d/fn
        if not p.exists(): continue
        rows = _read_csv(p)
        for r in rows:
            for k in list(r.keys()):
                if k.endswith("_id"): continue  # keep IDs so structure survives
                if random.random() < level:
                    r[k] = ""  # corrupted/missing
        _write_csv(p, rows)

def structural_noise(d, level):
    """Break a `level` fraction of cross-file references."""
    p = d/"mitigation_tracker.csv"
    if not p.exists(): return
    rows = _read_csv(p)
    for r in rows:
        if random.random() < level:
            r["linked_risk_id"] = "R999"  # dangling reference
    _write_csv(p, rows)

def inject_noise(src, dst, feat_level, struct_level):
    shutil.copytree(src, dst, dirs_exist_ok=True)
    feature_noise(Path(dst), feat_level)
    structural_noise(Path(dst), struct_level)
    return {"feature_noise":feat_level, "structural_noise":struct_level}

if __name__ == "__main__":
    # demo: 3 noise levels on one package
    for fl, sl in [(0.0,0.0),(0.3,0.1),(0.6,0.4)]:
        info = inject_noise("data/synthetic/base/PKG_C0000",
            f"data/synthetic/_noise_demo_f{fl}_s{sl}", fl, sl)
        print(f"  feat={fl} struct={sl} -> injected")
