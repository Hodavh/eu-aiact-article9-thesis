"""Inject a labelled Article 9 violation into a compliant package copy."""
import csv, json, shutil
from pathlib import Path
import random

def _read_csv(p): return list(csv.DictReader(open(p)))
def _write_csv(p, rows):
    with open(p,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
def _read_json(p): return json.load(open(p))
def _write_json(p,o): json.dump(o,open(p,"w"),indent=2)

# each returns the requirement it violates
def v_missing_mitigation(d):  # R6
    m=_read_csv(d/"mitigation_tracker.csv"); m.pop(); _write_csv(d/"mitigation_tracker.csv",m); return "R6"
def v_unacceptable_residual(d):  # R9
    m=_read_csv(d/"mitigation_tracker.csv"); m[0]["acceptability"]="not_acceptable"
    _write_csv(d/"mitigation_tracker.csv",m); return "R9"
def v_missing_rms_section(d):  # R1
    s=_read_json(d/"system_documentation.pdf.txt"); s[0]["rms_established"]="false"
    _write_json(d/"system_documentation.pdf.txt",s); return "R1"
def v_result_below_threshold(d):  # R15
    r=_read_json(d/"validation_test_results.json")
    r[0]["measured_value"]=round(float(r[0]["threshold"])-0.1,3); r[0]["pass_fail"]="fail"
    _write_json(d/"validation_test_results.json",r); return "R15"
def v_missing_postmarket(d):  # R5
    for f in ["model_drift_log.json","post_market_surveillance.csv","adverse_event_reports.pdf.txt"]:
        (d/f).unlink(missing_ok=True)
    return "R5"

VIOLATIONS = {"missing_mitigation":v_missing_mitigation, "unacceptable_residual":v_unacceptable_residual,
    "missing_rms":v_missing_rms_section, "result_below_threshold":v_result_below_threshold,
    "missing_postmarket":v_missing_postmarket}

def inject(src_dir, dst_dir, vtype):
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    req = VIOLATIONS[vtype](Path(dst_dir))
    return {"violation_type":vtype, "violated_requirement":req, "label":"non_compliant"}

if __name__ == "__main__":
    import sys
    for vt in VIOLATIONS:
        info = inject("data/synthetic/PKG_TEST", f"data/synthetic/PKG_{vt}", vt)
        print(f"  {vt}: violates {info['violated_requirement']}")
