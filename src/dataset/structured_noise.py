"""Structured, realistic noise model for compliance evidence.
Feature-side and structure-side modes are separated so channel asymmetry
can be controlled. Corruption is CORRELATED (whole records/columns/files),
not uniform random, matching real degradation (scanning, OCR, integration failures)."""
import csv, json, shutil, random
from pathlib import Path

CSV_FILES=["clinical_risk_register.csv","mitigation_tracker.csv",
           "clinical_validation_study.csv","traceability_matrix.csv"]
ID_COLS={"risk_id","mitigation_id","study_id","report_id","event_id","test_id","threat_id","scenario_id"}
# OCR-plausible substitutions (value becomes WRONG, not blank)
OCR_SUB={"serious":"serous","critical":"critcal","negligible":"negligable",
         "acceptable":"acceptabIe","true":"ture","high":"hlgh","1":"l","0":"O","5":"S"}

def _rc(p): return list(csv.DictReader(open(p))) if p.exists() else []
def _wc(p,rows):
    if not rows: return
    with open(p,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

# ---------- FEATURE-SIDE (hits the feature channel) ----------
def document_region_loss(d, level):
    """Whole RECORDS lose a correlated block of fields (scanned-region damage)."""
    for fn in CSV_FILES:
        rows=_rc(d/fn)
        for r in rows:
            if random.random()<level:
                keys=[k for k in r if k not in ID_COLS]
                # blank a contiguous block (a 'region'), not scattered cells
                if keys:
                    start=random.randint(0,len(keys)-1); span=random.randint(1,len(keys)-start)
                    for k in keys[start:start+span]: r[k]=""
        _wc(d/fn,rows)

def column_systematic_loss(d, level):
    """Whole FIELD TYPES degrade across all records (systematic OCR failure by column)."""
    for fn in CSV_FILES:
        rows=_rc(d/fn)
        if not rows: continue
        cols=[k for k in rows[0] if k not in ID_COLS]
        for c in cols:
            if random.random()<level:  # this column is a casualty
                for r in rows: r[c]=""
        _wc(d/fn,rows)

def ocr_substitution(d, level):
    """Values become WRONG (plausible but false), not blank. Harder to detect."""
    for fn in CSV_FILES:
        rows=_rc(d/fn)
        for r in rows:
            for k,v in r.items():
                if k in ID_COLS: continue
                if v in OCR_SUB and random.random()<level:
                    r[k]=OCR_SUB[v]
        _wc(d/fn,rows)

# ---------- STRUCTURE-SIDE (hits the structural channel) ----------
def file_integration_failure(d, level):
    """Integration failure breaks a FRACTION of a file's references (graded).
    At high level, most links dangle; at low level, few. Correlated (clustered)."""
    rows=_rc(d/"mitigation_tracker.csv")
    if not rows: return
    # clustered: pick a contiguous run of rows to break, size scaled by level
    n=len(rows); k=int(round(level*n))
    if k>0:
        start=random.randint(0,max(0,n-k))
        for r in rows[start:start+k]: r["linked_risk_id"]="R999"
    _wc(d/"mitigation_tracker.csv",rows)

def subtle_reference_corruption(d, level):
    """Links get typos/truncation (R001 -> R01), not obvious dead 'R999'."""
    rows=_rc(d/"mitigation_tracker.csv")
    for r in rows:
        rid=r.get("linked_risk_id","")
        if rid.startswith("R") and random.random()<level:
            r["linked_risk_id"]=rid[:2]+rid[3:] if len(rid)>3 else rid+"x"  # truncate/typo
    _wc(d/"mitigation_tracker.csv",rows)

# ---------- orchestrator ----------
def apply_structured_noise(src, dst, feature_level, structure_level):
    """feature_level, structure_level in [0,1]. Split across their sub-modes."""
    shutil.copytree(src,dst,dirs_exist_ok=True)
    d=Path(dst)
    # feature-side: three modes share the feature budget
    document_region_loss(d, feature_level*0.6)
    column_systematic_loss(d, feature_level*0.4)
    ocr_substitution(d, feature_level*0.5)
    # structure-side: two modes share the structure budget
    file_integration_failure(d, structure_level*0.7)
    subtle_reference_corruption(d, structure_level*0.8)
    return {"feature_level":feature_level,"structure_level":structure_level}

if __name__=="__main__":
    import sys; sys.path.insert(0,"src/layer3_gnn")
    from build_graph import estimate_noise
    print("Structured noise demo (measured feature, structural):")
    for fl,sl in [(0.0,0.0),(0.5,0.1),(0.1,0.5),(0.7,0.1)]:
        apply_structured_noise("data/synthetic/base/PKG_C0000","data/synthetic/_sn",fl,sl)
        print(f"  set f={fl} s={sl} -> measured {estimate_noise('data/synthetic/_sn')}")
        shutil.rmtree("data/synthetic/_sn",ignore_errors=True)
