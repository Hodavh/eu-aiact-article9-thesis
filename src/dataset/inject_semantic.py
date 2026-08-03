"""Inject labelled semantic manipulations (Layer 2 targets)."""
import csv, shutil, random
from pathlib import Path

def _read_csv(p): return list(csv.DictReader(open(p)))
def _write_csv(p,rows):
    with open(p,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

# a weak (information-level) description wrongly claimed as level 1 (elimination)
WEAK_DESC = ["warning shown in user manual about {h}", "note added to documentation about {h}",
             "user advised to check for {h}"]
# a serious harm wrongly recorded at low severity
SERIOUS_HARM = ["permanent vision loss from missed diagnosis", "patient death from delayed alert",
                "irreversible organ damage from wrong dose"]

def hierarchy_mismatch(d):
    p=d/"mitigation_tracker.csv"; rows=_read_csv(p)
    r=random.choice(rows)
    r["description"]=random.choice(WEAK_DESC).format(h="the hazard")
    r["hierarchy_level"]="1"  # claims elimination, but describes a mere warning
    _write_csv(p,rows)
    return {"type":"hierarchy_mismatch","mitigation_id":r["mitigation_id"]}

def severity_mismatch(d):
    p=d/"clinical_risk_register.csv"; rows=_read_csv(p)
    r=random.choice(rows)
    r["hazard"]=random.choice(SERIOUS_HARM)
    r["severity"]="negligible"  # serious harm claimed negligible
    _write_csv(p,rows)
    return {"type":"severity_mismatch","risk_id":r["risk_id"]}

def inject_semantic(src, dst, kind):
    shutil.copytree(src,dst,dirs_exist_ok=True)
    fn = hierarchy_mismatch if kind=="hierarchy" else severity_mismatch
    info = fn(Path(dst))
    info["label"]="semantic_violation"
    return info

if __name__ == "__main__":
    for k in ["hierarchy","severity"]:
        info=inject_semantic("data/synthetic/base/PKG_C0001",
            f"data/synthetic/_sem_demo_{k}", k)
        print(f"  {k}: {info}")
