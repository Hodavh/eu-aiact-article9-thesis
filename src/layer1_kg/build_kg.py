"""
Build the Article 9 regulatory knowledge graph in RDF/Turtle from the
gold standard requirements specification.

The CSV is the single source of truth; this script regenerates the graph.
"""

from pathlib import Path
import pandas as pd
from rdflib import Graph, Namespace, Literal, RDF, RDFS, XSD

# Paths
ROOT = Path(__file__).resolve().parents[2]
CSV_IN = ROOT / "data" / "gold_standard" / "requirements.csv"
TTL_OUT = ROOT / "data" / "gold_standard" / "article9_kg.ttl"

# Namespace
A9 = Namespace("http://eu-ai-act.eu/article9#")


def build_graph(df: pd.DataFrame) -> Graph:
    g = Graph()
    g.bind("a9", A9)

    # --- Ontology classes ---
    for cls, label in [
        (A9.Requirement, "Article 9 requirement"),
        (A9.EvidenceType, "Compliance evidence type"),
        (A9.ExternalReference, "Reference to another Article"),
    ]:
        g.add((cls, RDF.type, RDFS.Class))
        g.add((cls, RDFS.label, Literal(label)))

    evidence_seen = set()
    external_seen = set()

    for _, row in df.iterrows():
        req = A9[row["req_id"]]
        g.add((req, RDF.type, A9.Requirement))
        g.add((req, RDFS.label, Literal(row["req_id"])))

        # Provenance in the legal text
        g.add((req, A9.hasParagraph, Literal(int(row["paragraph"]), datatype=XSD.integer)))
        if pd.notna(row["subpoint"]):
            g.add((req, A9.hasSubpoint, Literal(str(row["subpoint"]))))

        # Core metadata
        g.add((req, A9.hasObligation, Literal(str(row["obligation"]))))
        g.add((req, A9.hasDeontic, Literal(str(row["deontic"]))))
        g.add((req, A9.hasRequirementType, Literal(str(row["req_type"]))))
        g.add((req, A9.hasCheckType, Literal(str(row["check_type"]))))
        g.add((req, A9.hasVerificationLogic, Literal(str(row["verification_logic"]))))
        g.add((req, A9.requiresSemanticCheck,
               Literal(bool(row["needs_semantic_check"]), datatype=XSD.boolean)))

        # Mitigation hierarchy level (Article 9(5)) - only R10, R11, R12
        if pd.notna(row["hierarchy_level"]):
            g.add((req, A9.hasHierarchyLevel,
                   Literal(int(row["hierarchy_level"]), datatype=XSD.integer)))

        # Cross-references
        if pd.notna(row["cross_refs"]):
            for ref in str(row["cross_refs"]).split(";"):
                ref = ref.strip()
                if not ref:
                    continue
                if ref.startswith("R") and ref[1:].isdigit():
                    g.add((req, A9.references, A9[ref]))
                else:  # external Article reference
                    node_id = ref.replace(" ", "_")
                    ext = A9[node_id]
                    if node_id not in external_seen:
                        g.add((ext, RDF.type, A9.ExternalReference))
                        g.add((ext, RDFS.label, Literal(ref)))
                        external_seen.add(node_id)
                    g.add((req, A9.references, ext))

        # Evidence type mappings
        if pd.notna(row["evidence_types"]):
            for ev in str(row["evidence_types"]).split(";"):
                ev = ev.strip()
                if not ev:
                    continue
                node_id = ev.replace(".", "_")
                ev_node = A9[node_id]
                if node_id not in evidence_seen:
                    g.add((ev_node, RDF.type, A9.EvidenceType))
                    g.add((ev_node, RDFS.label, Literal(ev)))
                    evidence_seen.add(node_id)
                g.add((req, A9.satisfiedBy, ev_node))

    return g


def main():
    df = pd.read_csv(CSV_IN)
    g = build_graph(df)
    g.serialize(destination=str(TTL_OUT), format="turtle")

    reqs = len(list(g.subjects(RDF.type, A9.Requirement)))
    evid = len(list(g.subjects(RDF.type, A9.EvidenceType)))
    exts = len(list(g.subjects(RDF.type, A9.ExternalReference)))
    triples = len(g)

    print(f"Wrote: {TTL_OUT}")
    print(f"  Requirement nodes:     {reqs}")
    print(f"  Evidence type nodes:   {evid}")
    print(f"  External references:   {exts}")
    print(f"  Total triples:         {triples}")


if __name__ == "__main__":
    main()
