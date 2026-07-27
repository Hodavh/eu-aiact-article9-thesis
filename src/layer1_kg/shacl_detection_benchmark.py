"""
SHACL detection-rate benchmark (RQ1).
Injects a range of known violations into copies of the gold-standard graph,
runs SHACL on each, and computes the detection rate.
"""
import json
from pathlib import Path
from rdflib import Graph, Namespace, Literal, XSD
from pyshacl import validate

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "gold_standard" / "article9_kg.ttl"
SHAPES = ROOT / "shapes" / "article9_shapes.ttl"
OUT = ROOT / "results" / "layer1_shacl_detection.json"
A9 = Namespace("http://eu-ai-act.eu/article9#")


def base_graph():
    return Graph().parse(str(GOLD), format="turtle")


def check(g):
    """Return True if SHACL flags a violation (i.e. non-conformant)."""
    conforms, _, _ = validate(g, shacl_graph=Graph().parse(str(SHAPES), format="turtle"),
                              inference="none", advanced=True)
    return not conforms  # detected == non-conformant


def build_violations():
    """Each entry: (name, description, mutation function). Mutation returns a graph."""
    v = []

    def missing_paragraph():
        g = base_graph(); g.remove((A9.R3, A9.hasParagraph, None)); return g
    v.append(("missing_paragraph", "R3 stripped of its paragraph", missing_paragraph))

    def missing_obligation():
        g = base_graph(); g.remove((A9.R4, A9.hasObligation, None)); return g
    v.append(("missing_obligation", "R4 stripped of obligation text", missing_obligation))

    def missing_deontic():
        g = base_graph(); g.remove((A9.R5, A9.hasDeontic, None)); return g
    v.append(("missing_deontic", "R5 stripped of deontic modality", missing_deontic))

    def missing_reqtype():
        g = base_graph(); g.remove((A9.R6, A9.hasRequirementType, None)); return g
    v.append(("missing_reqtype", "R6 stripped of requirement type", missing_reqtype))

    def missing_semantic_flag():
        g = base_graph(); g.remove((A9.R8, A9.requiresSemanticCheck, None)); return g
    v.append(("missing_semantic_flag", "R8 stripped of semantic-check flag", missing_semantic_flag))

    def bad_deontic_value():
        g = base_graph(); g.remove((A9.R7, A9.hasDeontic, None))
        g.add((A9.R7, A9.hasDeontic, Literal("MUST"))); return g
    v.append(("bad_deontic_value", "R7 deontic set to illegal MUST", bad_deontic_value))

    def bad_reqtype_value():
        g = base_graph(); g.remove((A9.R9, A9.hasRequirementType, None))
        g.add((A9.R9, A9.hasRequirementType, Literal("optional"))); return g
    v.append(("bad_reqtype_value", "R9 requirement type set to illegal 'optional'", bad_reqtype_value))

    def paragraph_out_of_range():
        g = base_graph(); g.remove((A9.R1, A9.hasParagraph, None))
        g.add((A9.R1, A9.hasParagraph, Literal(15, datatype=XSD.integer))); return g
    v.append(("paragraph_out_of_range", "R1 paragraph set to 15 (>10)", paragraph_out_of_range))

    def hierarchy_out_of_range():
        g = base_graph(); g.remove((A9.R11, A9.hasHierarchyLevel, None))
        g.add((A9.R11, A9.hasHierarchyLevel, Literal(5, datatype=XSD.integer))); return g
    v.append(("hierarchy_out_of_range", "R11 hierarchy level set to 5", hierarchy_out_of_range))

    def verifiable_no_evidence():
        g = base_graph(); g.remove((A9.R13, A9.satisfiedBy, None)); return g
    v.append(("verifiable_no_evidence", "R13 (verifiable) stripped of all evidence links", verifiable_no_evidence))

    return v


def main():
    violations = build_violations()
    results = []
    detected = 0
    for name, desc, mutate in violations:
        g = mutate()
        caught = check(g)
        detected += int(caught)
        results.append({"violation": name, "description": desc, "detected": caught})
        print(f"  [{'CAUGHT ' if caught else 'MISSED '}] {name}: {desc}")

    n = len(violations)
    rate = detected / n
    summary = {
        "total_violations_tested": n,
        "detected": detected,
        "missed": n - detected,
        "detection_rate": round(rate, 3),
        "violations": results,
    }
    json.dump(summary, open(OUT, "w"), indent=2)

    print()
    print("=" * 50)
    print(f"SHACL DETECTION RATE: {detected}/{n} = {rate:.3f}")
    print("=" * 50)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
