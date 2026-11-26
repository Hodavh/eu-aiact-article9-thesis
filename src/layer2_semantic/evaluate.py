"""Layer 2 evaluation: detection rate, precision, recall, F1 on planted manipulations."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"dataset"))
from verify import check_hierarchy, check_severity
from inject_semantic import inject_semantic
import csv, shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/"data"/"synthetic"/"base"
OUT = ROOT/"results"/"layer2_metrics.json"

N = 30  # manipulated packages per type + clean controls

def has_mismatch_hier(pkg):
    """True if Layer 2 flags any hierarchy MISMATCH in the package."""
    for r in csv.DictReader(open(Path(pkg)/"mitigation_tracker.csv")):
        if check_hierarchy(r["description"], r["hierarchy_level"])["flag"]=="MISMATCH":
            return True
    return False

def has_mismatch_sev_planted(pkg, planted_risk_id):
    """Check specifically whether the planted severity record is flagged."""
    for r in csv.DictReader(open(Path(pkg)/"clinical_risk_register.csv")):
        if r["risk_id"]==planted_risk_id:
            return check_severity(r["hazard"], r["severity"])["flag"]=="MISMATCH"
    return False

def main():
    tmp = ROOT/"data"/"synthetic"/"_l2eval"
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir()

    tp=fp=tn=fn=0
    pkgs = sorted([d for d in BASE.iterdir() if d.name.startswith("PKG_C")])[:N*2]

    # POSITIVES: inject a hierarchy manipulation, expect detection
    for i,p in enumerate(pkgs[:N]):
        dst=tmp/f"pos_{i}"
        info=inject_semantic(p, dst, "hierarchy")
        detected = has_mismatch_hier(dst)
        if detected: tp+=1
        else: fn+=1

    # NEGATIVES: clean packages, expect NO hierarchy mismatch
    for i,p in enumerate(pkgs[N:N*2]):
        detected = has_mismatch_hier(p)
        if detected: fp+=1
        else: tn+=1

    prec = tp/(tp+fp) if tp+fp else 0
    rec = tp/(tp+fn) if tp+fn else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    det_rate = tp/N

    res = {"n_positive":N,"n_negative":N,"tp":tp,"fp":fp,"tn":tn,"fn":fn,
           "detection_rate":round(det_rate,3),"precision":round(prec,3),
           "recall":round(rec,3),"f1":round(f1,3)}
    json.dump(res, open(OUT,"w"), indent=2)
    shutil.rmtree(tmp)

    print("=== LAYER 2 EVALUATION (hierarchy manipulation) ===")
    print(f"Positives (manipulated): {N}  | Negatives (clean): {N}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Detection rate: {det_rate:.3f}")
    print(f"Precision:      {prec:.3f}")
    print(f"Recall:         {rec:.3f}")
    print(f"F1:             {f1:.3f}")
    print(f"Saved: {OUT}")

if __name__=="__main__":
    main()
