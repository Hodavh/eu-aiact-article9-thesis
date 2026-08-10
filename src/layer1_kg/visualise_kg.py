"""Render the Article 9 knowledge graph: requirement structure by paragraph."""

from pathlib import Path
from rdflib import Graph, Namespace, RDF
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
TTL = ROOT / "data" / "gold_standard" / "article9_kg.ttl"
OUT = ROOT / "results" / "article9_kg_requirements.png"
A9 = Namespace("http://eu-ai-act.eu/article9#")


def short(u):
    return str(u).split("#")[-1]


def load():
    g = Graph().parse(str(TTL), format="turtle")
    reqs = {}
    for s in g.subjects(RDF.type, A9.Requirement):
        rid = short(s)
        reqs[rid] = dict(
            para=int(next(g.objects(s, A9.hasParagraph))),
            sub=str(next(g.objects(s, A9.hasSubpoint), "")),
            semantic=bool(next(g.objects(s, A9.requiresSemanticCheck))),
            rtype=str(next(g.objects(s, A9.hasRequirementType))),
            level=next(g.objects(s, A9.hasHierarchyLevel), None),
        )
    refs = [(short(s), short(o)) for s, o in g.subject_objects(A9.references)
            if short(o).startswith("R") and short(o)[1:].isdigit()]
    return reqs, refs


def main():
    reqs, refs = load()
    G = nx.DiGraph()
    G.add_nodes_from(reqs)
    G.add_edges_from(refs)

    by_para = {}
    for r, d in reqs.items():
        by_para.setdefault(d["para"], []).append(r)

    # top-aligned stacks, tight vertical spacing
    pos = {}
    for para in sorted(by_para):
        for i, r in enumerate(sorted(by_para[para], key=lambda x: int(x[1:]))):
            pos[r] = (para * 2.2, -i * 1.35)

    fig, ax = plt.subplots(figsize=(19, 7.5))

    colours = []
    for r in G.nodes():
        d = reqs[r]
        if d["rtype"] == "structural_constraint":
            colours.append("#94A3B8")
        elif d["rtype"] == "permissive_provision":
            colours.append("#CBD5E1")
        elif d["semantic"]:
            colours.append("#F97316")
        else:
            colours.append("#2563EB")

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#94A3B8", width=1.3,
                           arrowsize=13, alpha=0.65, node_size=2300,
                           connectionstyle="arc3,rad=0.18")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colours,
                           edgecolors="#0F172A", linewidths=1.3, node_size=2300)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10.5,
                            font_weight="bold", font_color="white")

    # paragraph headers above the columns
    top = max(y for _, y in pos.values())
    for para in sorted(by_para):
        ax.text(para * 2.2, top + 1.0, f"¶{para}", ha="center",
                fontsize=13, fontweight="bold", color="#0F172A")

    # citation labels under each node
    for r, (x, y) in pos.items():
        d = reqs[r]
        cite = f"9({d['para']})" + (f"({d['sub']})" if d["sub"] and d["sub"] != "chapeau" else "")
        if d["level"]:
            cite += f" · level {int(d['level'])}"
        ax.text(x, y - 0.58, cite, ha="center", fontsize=8, color="#475569")

    legend = [
        mpatches.Patch(color="#2563EB", label="Verifiable — structural check (Layer 1)"),
        mpatches.Patch(color="#F97316", label="Verifiable — semantic check (Layer 2)"),
        mpatches.Patch(color="#94A3B8", label="Structural constraint (not scored)"),
        mpatches.Patch(color="#CBD5E1", label="Permissive provision (not scored)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9.5,
              frameon=True, framealpha=0.95, bbox_to_anchor=(1.0, 1.0))

    ax.set_xlim(1.2, 23.2)
    ax.set_ylim(min(y for _, y in pos.values()) - 1.3, top + 1.7)
    ax.axis("off")
    fig.suptitle("EU AI Act Article 9: Requirement Structure and Cross-References",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
