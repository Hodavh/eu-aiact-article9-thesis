"""
Validate an Article 9 knowledge graph against the SHACL shapes.
Usage: python validate_kg.py [path_to_graph.ttl]
Defaults to the gold standard graph.
"""
import sys
from pathlib import Path
from pyshacl import validate
from rdflib import Graph

ROOT = Path(__file__).resolve().parents[2]
SHAPES = ROOT / "shapes" / "article9_shapes.ttl"
DEFAULT_DATA = ROOT / "data" / "gold_standard" / "article9_kg.ttl"


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    data = Graph().parse(str(data_path), format="turtle")
    shapes = Graph().parse(str(SHAPES), format="turtle")

    conforms, report_graph, report_text = validate(
        data, shacl_graph=shapes,
        inference="none", advanced=True, meta_shacl=False,
    )

    print(f"Validating: {data_path.name}")
    print(f"Conforms: {conforms}")
    if not conforms:
        print("\n--- Violations ---")
        print(report_text)
    else:
        print("All SHACL constraints satisfied.")
    return 0 if conforms else 1


if __name__ == "__main__":
    sys.exit(main())
