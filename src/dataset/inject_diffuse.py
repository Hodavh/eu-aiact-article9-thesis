"""Diffuse multi-signal violations: non-compliance emerges from MANY weak defects
spread across records, not one clean flip. Forces aggregation across the graph."""
import csv, shutil, random
from pathlib import Path

def _rc(p): return list(csv.DictReader(open(p)))
def _wc(p,rows):
    if not rows: return
    with open(p,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def inject_diffuse(src, dst, severity=0.5):
    """severity in [0,1]: fraction of records given a weak defect.
    A package is non-compliant if the ACCUMULATED weakness crosses a threshold."""
    shutil.copytree(src,dst,dirs_exist_ok=True)
    d=Path(dst)
    risks=_rc(d/"clinical_risk_register.csv")
    mits=_rc(d/"mitigation_tracker.csv")
    defect_count=0

    # spread weak defects across risk records (feature-side signal)
    for r in risks:
        if random.random()<severity:
            # weaken: borderline severity understatement (not a gross flip)
            order=["negligible","minor","serious","critical","catastrophic"]
            if r["severity"] in order:
                i=order.index(r["severity"])
                if i>0: r["severity"]=order[i-1]; defect_count+=1  # one notch down

    # spread weak defects across mitigation records (feature + structure side)
    for m in mits:
        if random.random()<severity:
            roll=random.random()
            if roll<0.4:
                m["residual_risk"]="high"; defect_count+=1          # feature-side
            elif roll<0.7 and m["acceptability"]=="acceptable":
                m["acceptability"]="not_acceptable"; defect_count+=1  # feature-side
            else:
                m["linked_risk_id"]="R999"; defect_count+=1          # structure-side (broken link)

    _wc(d/"clinical_risk_register.csv",risks)
    _wc(d/"mitigation_tracker.csv",mits)

    # label: non-compliant if accumulated defects exceed threshold (needs aggregation to detect)
    total=len(risks)+len(mits)
    ratio=defect_count/total if total else 0
    label="non_compliant" if ratio>=0.25 else "compliant"
    return {"label":label,"defect_ratio":round(ratio,3),"defect_count":defect_count}

if __name__=="__main__":
    # demo across severities
    for sev in [0.0,0.2,0.5,0.8]:
        info=inject_diffuse("data/synthetic/base/PKG_C0000",
            f"data/synthetic/_diff_{sev}", severity=sev)
        print(f"  severity={sev}: {info}")
    import shutil
    for sev in [0.0,0.2,0.5,0.8]: shutil.rmtree(f"data/synthetic/_diff_{sev}",ignore_errors=True)
