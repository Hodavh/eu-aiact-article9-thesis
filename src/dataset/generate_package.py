"""Generate one compliant Article 9 evidence package."""
import json, random, csv
from pathlib import Path
from datetime import date, timedelta
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import (SEVERITY, LIKELIHOOD, AFFECTED_DIMENSION, AFFECTED_GROUP,
                     RESIDUAL_RISK, ACCEPTABILITY, PERF)

HAZARDS = ["false negative diagnosis", "false positive diagnosis", "image quality failure",
    "model drift over time", "bias against subgroup", "automation over-reliance",
    "incorrect risk score", "delayed alert", "data input error", "edge-case misclassification"]
METRICS = ["sensitivity", "specificity", "auc", "ppv", "npv"]
HIER_DESC = {
    1: ["redesigned model architecture to eliminate {h}", "improved training data to remove {h}"],
    2: ["added confidence threshold to control {h}", "implemented human-in-loop check for {h}"],
    3: ["warning in user manual about {h}", "training provided to deployers on {h}"],
}

def _rng(lo, hi): return round(random.uniform(lo, hi), 3)

def make_package(pkg_id, out_dir, n_risks=8):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # risk register
    risks = []
    for i in range(n_risks):
        h = random.choice(HAZARDS)
        risks.append(dict(risk_id=f"R{i+1:03d}", hazard=h,
            affected_dimension=random.choice(AFFECTED_DIMENSION),
            likelihood=random.choice(LIKELIHOOD),
            severity=random.choice(SEVERITY),
            intended_use_context="clinical decision support",
            mitigable="true",
            affected_group=random.choice(AFFECTED_GROUP)))
    _write_csv(out/"clinical_risk_register.csv", risks)

    # mitigation tracker: every risk linked, valid hierarchy
    mits = []
    for i, r in enumerate(risks):
        lvl = random.choice([1, 2, 3])
        desc = random.choice(HIER_DESC[lvl]).format(h=r["hazard"])
        mits.append(dict(mitigation_id=f"M{i+1:03d}", linked_risk_id=r["risk_id"],
            hierarchy_level=lvl, description=desc,
            residual_risk=random.choice(RESIDUAL_RISK),
            acceptability="acceptable"))
    _write_csv(out/"mitigation_tracker.csv", mits)

    # validation protocol + results (shared metrics/thresholds)
    proto, results = [], []
    for m in random.sample(METRICS, 3):
        thr = _rng(0.80, 0.90)
        proto.append(dict(metric_name=m, threshold=thr, threshold_direction="min"))
        meas = _rng(thr, 0.97)
        results.append(dict(metric_name=m, threshold=thr, measured_value=meas,
            pass_fail="pass" if meas >= thr else "fail"))
    _write_csv(out/"validation_protocol.pdf.csv", proto)
    _write_json(out/"validation_test_results.json", results)

    # clinical validation study
    study = [dict(study_id="S001", metric_name=m,
        measured_value=_rng(*PERF[m]) if m in PERF else _rng(0.85,0.95),
        sample_size=random.randint(*PERF["sample_size"]), outcome="met_endpoint")
        for m in random.sample(METRICS, 2)]
    _write_csv(out/"clinical_validation_study.csv", study)

    # post-market (R5): at least one present
    _write_json(out/"model_drift_log.json", [dict(timestamp=str(date.today()),
        metric_name="auc", baseline_value=0.94, current_value=0.93, drift_detected="false")])
    _write_csv(out/"post_market_surveillance.csv", [dict(report_id="PMS001",
        period="2025-Q1", issues_identified="none", action_taken="routine review")])
    _write_pdf_txt(out/"adverse_event_reports.pdf.txt",
        [dict(event_id="AE001", event_date=str(date.today()), description="minor usability issue",
              severity="minor", linked_risk_id="R001")])

    # process/lifecycle
    _write_pdf_txt(out/"system_documentation.pdf.txt", [dict(section="risk_management_system",
        content="RMS established per Article 9", intended_purpose="clinical decision support",
        rms_established="true")])
    _write_json(out/"retraining_documentation.json", [dict(event_id="RT001",
        event_date=str(date.today()-timedelta(days=90)), trigger="scheduled review", outcome="model updated")])
    _write_json(out/"regression_testing_record.json", [dict(test_id="RG001",
        test_date=str(date.today()), trigger="post-retrain", result="pass")])

    # risk analysis (R4)
    _write_json(out/"threat_model.json", [dict(threat_id="T001",
        misuse_scenario="use outside intended population", likelihood="remote", linked_risk_id="R001")])
    _write_json(out/"misuse_scenario_analysis.json", [dict(scenario_id="MS001",
        misuse_scenario="ignoring low-confidence output", consequence="missed diagnosis", likelihood="occasional")])

    # traceability (R8)
    _write_csv(out/"traceability_matrix.csv", [dict(mitigation_id=m["mitigation_id"],
        linked_requirement="R6", linked_test_id="RG001") for m in mits])

    return pkg_id

def _write_csv(p, rows):
    if not rows: return
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
def _write_json(p, obj): json.dump(obj, open(p, "w"), indent=2)
def _write_pdf_txt(p, rows): json.dump(rows, open(p, "w"), indent=2)

if __name__ == "__main__":
    pid = make_package("PKG_TEST", "data/synthetic/PKG_TEST")
    print(f"Generated compliant package: {pid}")
    import os
    for f in sorted(os.listdir("data/synthetic/PKG_TEST")):
        print(f"  {f}")
