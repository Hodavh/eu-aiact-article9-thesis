"""
Layer 1 extraction evaluation (RQ1).
Compares LLM-extracted requirements against the gold standard,
distinguishing genuine errors from legitimate decomposition-granularity
differences.
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ALIGN = ROOT / "data" / "gold_standard" / "llm_alignment.csv"
GOLD = ROOT / "data" / "gold_standard" / "requirements.csv"
OUT = ROOT / "results" / "layer1_extraction_metrics.json"


def main():
    align = pd.read_csv(ALIGN)
    gold = pd.read_csv(GOLD)

    gold_ids = set(gold["req_id"])
    n_gold = len(gold_ids)
    n_extracted = len(align)

    # gold requirements covered by at least one extracted item
    covered = set(align["maps_to"]) & gold_ids
    completeness = len(covered) / n_gold

    # category counts
    cats = align["category"].value_counts().to_dict()
    n_match = cats.get("match", 0)
    n_decomp = cats.get("decomposition", 0)
    n_dup = cats.get("duplicate", 0)
    n_deontic = cats.get("deontic_error", 0)
    n_spurious = cats.get("spurious", 0)

    genuine_errors = n_dup + n_deontic + n_spurious

    # strict: only one-to-one matches count as correct
    precision_strict = n_match / n_extracted
    # adjusted: decomposition differences accepted; only genuine errors penalised
    precision_adjusted = (n_extracted - genuine_errors) / n_extracted

    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0

    results = {
        "gold_requirements": n_gold,
        "extracted_items": n_extracted,
        "gold_covered": len(covered),
        "missing": sorted(gold_ids - covered),
        "completeness_recall": round(completeness, 3),
        "category_breakdown": {
            "match": n_match,
            "decomposition_granularity": n_decomp,
            "duplicate_error": n_dup,
            "deontic_error": n_deontic,
            "spurious_error": n_spurious,
        },
        "genuine_errors": genuine_errors,
        "precision_strict": round(precision_strict, 3),
        "precision_adjusted": round(precision_adjusted, 3),
        "f1_strict": round(f1(precision_strict, completeness), 3),
        "f1_adjusted": round(f1(precision_adjusted, completeness), 3),
    }

    json.dump(results, open(OUT, "w"), indent=2)

    print("=" * 55)
    print("LAYER 1 EXTRACTION EVALUATION (RQ1)")
    print("=" * 55)
    print(f"Gold-standard requirements:     {n_gold}")
    print(f"LLM-extracted items:            {n_extracted}")
    print(f"Gold requirements covered:      {len(covered)}/{n_gold}")
    print(f"Missing (recall failures):      {results['missing'] or 'none'}")
    print()
    print(f"Completeness (recall):          {completeness:.3f}")
    print()
    print("Extracted-item categories:")
    print(f"  Exact matches:                {n_match}")
    print(f"  Decomposition-granularity:    {n_decomp}  (legitimate, not errors)")
    print(f"  Duplicate errors:             {n_dup}")
    print(f"  Deontic errors:               {n_deontic}")
    print(f"  Spurious errors:              {n_spurious}")
    print(f"  Genuine errors total:         {genuine_errors}")
    print()
    print(f"Precision (strict):             {precision_strict:.3f}")
    print(f"Precision (adjusted):           {precision_adjusted:.3f}")
    print(f"F1 (strict):                    {f1(precision_strict, completeness):.3f}")
    print(f"F1 (adjusted):                  {f1(precision_adjusted, completeness):.3f}")
    print()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
