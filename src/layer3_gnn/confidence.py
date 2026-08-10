"""Add confidence scoring and deferral to the trained GNN."""
import torch
import torch.nn.functional as F

def predict_with_confidence(model, data, threshold=0.75):
    """Return (prediction, confidence, decision).
    decision = 'accept' if confident, else 'defer' to human review."""
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
    conf = conf.item(); pred = pred.item()
    decision = "accept" if conf >= threshold else "defer"
    return {"prediction": "non_compliant" if pred==1 else "compliant",
            "confidence": round(conf,3), "decision": decision}

if __name__=="__main__":
    import sys, json, random
    from pathlib import Path
    sys.path.insert(0,"src/layer3_gnn"); sys.path.insert(0,"src/dataset")
    from build_graph import build_graph
    from model import DualShieldGNN
    from torch_geometric.loader import DataLoader

    ROOT=Path(".")
    LABELS=json.load(open("data/synthetic/labels.json"))
    BASE=Path("data/synthetic/base")

    # quick-train a model to demo confidence
    random.seed(0)
    data=[]
    for pid,info in list(LABELS.items()):
        g=build_graph(BASE/pid); g.y=torch.tensor([0 if info["label"]=="compliant" else 1]); data.append(g)
    random.shuffle(data); sp=int(0.8*len(data)); tr,te=data[:sp],data[sp:]
    trl=DataLoader(tr,batch_size=32,shuffle=True)
    m=DualShieldGNN(hid=48,fusion="noise_aware")
    opt=torch.optim.Adam(m.parameters(),lr=0.002,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=120)
    lf=torch.nn.CrossEntropyLoss()
    for _ in range(120):
        m.train()
        for b in trl: opt.zero_grad(); lf(m(b),b.y).backward(); opt.step()
        sched.step()

    # show confidence + deferral on 12 test packages
    accept=defer=0
    print("Prediction / confidence / decision:")
    for g in te[:12]:
        r=predict_with_confidence(m, g)
        print(f"  {r['prediction']:14} conf={r['confidence']:.3f} -> {r['decision']}")
    for g in te:
        r=predict_with_confidence(m,g)
        accept += r["decision"]=="accept"; defer += r["decision"]=="defer"
    print(f"\nOver {len(te)} test packages: {accept} accepted, {defer} deferred to human review")
