"""Add per-requirement ground truth to labels."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "synthetic" / "labels.json"

VERIFIABLE = ["R1","R2","R3","R4","R5","R6","R8","R9","R10","R11","R12","R13","R14","R15","R16"]

def main():
    labels = json.load(open(LABELS))
    for pid, info in labels.items():
        violated = info.get("violated_requirement")
        # every verifiable req passes except the violated one
        info["requirement_status"] = {
            r: ("fail" if r == violated else "pass") for r in VERIFIABLE
        }
    json.dump(labels, open(LABELS,"w"), indent=2)
    # sanity
    n_fail = sum(1 for v in labels.values()
                 for s in v["requirement_status"].values() if s=="fail")
    print(f"Updated {len(labels)} packages with per-requirement labels")
    print(f"Total requirement-level failures: {n_fail} (expect 100, one per non-compliant pkg)")

if __name__ == "__main__":
    main()
