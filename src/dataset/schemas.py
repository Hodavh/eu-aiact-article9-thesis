"""Evidence-file schemas for synthetic compliance packages.
Fields driven by Article 9 verification logic; value ranges grounded in
ISO 14971 (severity/probability) and IDx-DR DEN180001 (performance)."""

SEVERITY = ["negligible", "minor", "serious", "critical", "catastrophic"]
LIKELIHOOD = ["improbable", "remote", "occasional", "probable", "frequent"]
AFFECTED_DIMENSION = ["health", "safety", "fundamental_rights"]
AFFECTED_GROUP = ["general", "under_18", "vulnerable"]
RESIDUAL_RISK = ["low", "medium", "high"]
ACCEPTABILITY = ["acceptable", "not_acceptable"]
HIERARCHY_LEVEL = [1, 2, 3]

PERF = dict(sensitivity=(0.85, 0.95), specificity=(0.85, 0.95),
            auc=(0.90, 0.97), sample_size=(500, 2000))

SCHEMAS = {
    "clinical_risk_register.csv": ["risk_id", "hazard", "affected_dimension",
        "likelihood", "severity", "intended_use_context", "mitigable", "affected_group"],
    "mitigation_tracker.csv": ["mitigation_id", "linked_risk_id", "hierarchy_level",
        "description", "residual_risk", "acceptability"],
    "validation_protocol.pdf": ["metric_name", "threshold", "threshold_direction"],
    "validation_test_results.json": ["metric_name", "threshold", "measured_value", "pass_fail"],
    "clinical_validation_study.csv": ["study_id", "metric_name", "measured_value",
        "sample_size", "outcome"],
    "adverse_event_reports.pdf": ["event_id", "event_date", "description", "severity", "linked_risk_id"],
    "model_drift_log.json": ["timestamp", "metric_name", "baseline_value", "current_value", "drift_detected"],
    "post_market_surveillance.csv": ["report_id", "period", "issues_identified", "action_taken"],
    "system_documentation.pdf": ["section", "content", "intended_purpose", "rms_established"],
    "retraining_documentation.json": ["event_id", "event_date", "trigger", "outcome"],
    "regression_testing_record.json": ["test_id", "test_date", "trigger", "result"],
    "threat_model.json": ["threat_id", "misuse_scenario", "likelihood", "linked_risk_id"],
    "misuse_scenario_analysis.json": ["scenario_id", "misuse_scenario", "consequence", "likelihood"],
    "traceability_matrix.csv": ["mitigation_id", "linked_requirement", "linked_test_id"],
}

if __name__ == "__main__":
    print(f"{len(SCHEMAS)} evidence-file schemas defined")
    for f, cols in SCHEMAS.items():
        print(f"  {f}: {len(cols)} fields")
