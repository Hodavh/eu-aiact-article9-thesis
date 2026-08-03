"""Validate integrity of the base synthetic dataset."""
import json, csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import SCHEMAS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "synthetic" / "base"
LABELS = ROOT / "data" / "synthetic" / "labels.json"

# expected files (pdf placeholders written as .pdf.txt / .pdf.csv)
EXPECTED = set()
for f in SCHEMAS:
    if f.endswith(".pdf"):
        EXPECTED.add(f + ".txt" if "system" in f or "adverse" in f else f + ".csv")
    else:
        EXPECTED.add(f)

def main():
    labels = json.load(open(LABELS))
    pkgs = [d for d in BASE.iterdir() if d.is_dir() and d.name.startswith("PKG")]
    print(f"Packages on disk: {len(pkgs)}  | Labels: {len(labels)}")

    missing_files, label_mismatch, empty_files = 0, 0, 0
    for p in pkgs:
        if p.name not in labels: label_mismatch += 1
        present = {f.name for f in p.iterdir()}
        # non-compliant 'missing_postmarket' legitimately drops 3 files
        vt = labels.get(p.name,{}).get("violation_type")
        exp = set(EXPECTED)
        if vt == "missing_postmarket":
            exp -= {"model_drift_log.json","post_market_surveillance.csv","adverse_event_reports.pdf.txt"}
        miss = exp - present
        if miss: missing_files += 1
        for f in p.iterdir():
            if f.stat().st_size == 0: empty_files += 1

    print(f"Packages missing expected files: {missing_files}")
    print(f"Packages not in labels:          {label_mismatch}")
    print(f"Empty files found:               {empty_files}")
    print("INTEGRITY:", "PASS" if missing_files==0 and label_mismatch==0 and empty_files==0 else "CHECK")

if __name__ == "__main__":
    main()
