"""End-to-end compliance verification pipeline: Layer 1 -> Layer 2 -> Layer 3."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent/"layer2_semantic"))
sys.path.insert(0, str(Path(__file__).resolve().parent/"layer3_gnn"))
from verify import verify_package          # Layer 2
from build_graph import build_graph        # Layer 3 graph
from model import DualShieldGNN            # Layer 3 model
from confidence import predict_with_confidence

ROOT = Path(__file__).resolve().parent.parent

def verify_compliance(pkg_dir, model, run_semantic=True, threshold=0.75):
    """Full pipeline verdict for one package.
    Layer 1 (structural) is assumed run at dataset build (SHACL on the KG);
    here we run Layer 2 semantic flags and Layer 3 classification, wired together."""
    pkg = Path(pkg_dir)
    report = {"package": pkg.name, "layers": {}}

    # --- Layer 2: semantic verification (produces flags) ---
    semantic_flags = []
    if run_semantic:
        semantic_flags = verify_package(pkg)
        mismatches = [f for f in semantic_flags if f.get("flag") == "MISMATCH"]
        report["layers"]["layer2_semantic"] = {
            "checks_run": len(semantic_flags),
            "mismatches": len(mismatches),
            "flagged_records": [f["record_id"] for f in mismatches],
        }

    # --- Layer 3: GNN classification, WITH Layer 2 flags as node features (task 7.11) ---
    graph = build_graph(pkg, semantic_flags=semantic_flags)
    result = predict_with_confidence(model, graph, threshold=threshold)
    report["layers"]["layer3_classification"] = result
    report["measured_noise"] = {"feature": round(graph.noise[0].item(),3),
                                 "structural": round(graph.noise[1].item(),3)}

    # --- Combined verdict (conservative escalation, matches framework design) ---
    l2_mismatches = report["layers"]["layer2_semantic"]["mismatches"] if run_semantic else 0
    l3_noncompliant = result["prediction"] == "non_compliant"
    l3_confident = result["decision"] == "accept"

    # Decision policy:
    #  - Strong evidence of non-compliance (Layer 3 confident-noncompliant,
    #    OR multiple corroborating semantic mismatches) -> NON_COMPLIANT
    #  - Any weaker signal (a single semantic flag, or low Layer 3 confidence) -> REVIEW
    #  - Otherwise -> COMPLIANT
    # Division of labour:
    #  - Layer 3 (trained classifier) drives automatic verdicts when confident.
    #  - Layer 2 semantic flags are REVIEW TRIGGERS, not auto-rejections
    #    (its high-recall/moderate-precision profile suits surfacing, not deciding).
    if not l3_confident:
        verdict = "REVIEW"                       # Layer 3 unsure -> human
    elif l3_noncompliant:
        verdict = "NON_COMPLIANT"                # Layer 3 confident it fails
    elif l2_mismatches >= 1:
        verdict = "REVIEW"                       # Layer 3 says ok, but Layer 2 raised a concern -> human checks
    else:
        verdict = "COMPLIANT"                    # both clear
    report["verdict"] = verdict
    report["reasoning"] = {
        "layer2_mismatches": l2_mismatches,
        "layer3_prediction": result["prediction"],
        "layer3_confident": l3_confident,
        "layer3_confidence": result["confidence"],
        "decision_basis": ("strong non-compliance signal" if verdict=="NON_COMPLIANT"
                           else "uncertain: escalated to human review" if verdict=="REVIEW"
                           else "passed all checks with confidence"),
    }
    return report

def _quick_train():
    """Train a Layer 3 model on clean base data for the demo."""
    import random
    from torch_geometric.loader import DataLoader
    LABELS=json.load(open(ROOT/"data"/"synthetic"/"labels.json"))
    BASE=ROOT/"data"/"synthetic"/"base"
    import torch as _t; _t.manual_seed(1); random.seed(0); data=[]
    for pid,info in LABELS.items():
        g=build_graph(BASE/pid); g.y=torch.tensor([0 if info["label"]=="compliant" else 1]); data.append(g)
    random.shuffle(data); tr=data[:int(0.8*len(data))]
    trl=DataLoader(tr,batch_size=32,shuffle=True)
    m=DualShieldGNN(hid=48,fusion="noise_aware")
    opt=torch.optim.Adam(m.parameters(),lr=0.002,weight_decay=5e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=120)
    lf=torch.nn.CrossEntropyLoss()
    for _ in range(150):
        m.train()
        for b in trl: opt.zero_grad(); lf(m(b),b.y).backward(); opt.step()
        sch.step()
    return m

if __name__=="__main__":
    print("Training Layer 3 model for pipeline demo...")
    model=_quick_train()
    print("Running end-to-end pipeline on sample packages:\n")
    # one compliant, one non-compliant
    for pkg in ["data/synthetic/base/PKG_C0000","data/synthetic/base/PKG_N0000"]:
        rep=verify_compliance(ROOT/pkg, model, run_semantic=True)
        print(f"=== {rep['package']} ===")
        print(f"  VERDICT: {rep['verdict']}")
        print(f"  Layer 2: {rep['layers']['layer2_semantic']['mismatches']} semantic mismatch(es)")
        print(f"  Layer 3: {rep['layers']['layer3_classification']['prediction']} "
              f"(conf {rep['layers']['layer3_classification']['confidence']})")
        print(f"  Noise: {rep['measured_noise']}")
        print(f"  Basis: {rep['reasoning']['decision_basis']}\n")
