"""Sec. 4.3 (clean data): 5-fold cross-validation on the clean single-violation base dataset.

Confirms that the compliance signal can be learned from the graph representation before any
noise is added. Uses the same settings as every other Layer 3 experiment in the paper
(Adam, lr 0.002, weight decay 5e-4, cosine annealing over 120 epochs, batch size 32,
48 hidden units) and a fixed shuffle, so all fusion strategies see identical folds.

Usage: python src/layer3_gnn/experiment_cleancv.py [repo_root]
"""
import json, random, sys
from pathlib import Path
import numpy as np, torch
from torch_geometric.loader import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_graph import build_graph
from model import DualShieldGNN

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "synthetic" / "base"
LABELS = json.load(open(ROOT / "data" / "synthetic" / "labels.json"))
K = 5

def load():
    data = []
    for pid, info in LABELS.items():
        g = build_graph(BASE / pid)
        g.y = torch.tensor([0 if info["label"] == "compliant" else 1])
        data.append(g)
    random.Random(123).shuffle(data)  # fixed shuffle -> identical folds for all methods
    return data

def folds(data, k=K):
    size = len(data) // k
    for i in range(k):
        yield data[:i*size] + data[(i+1)*size:], data[i*size:(i+1)*size]

def train_eval(fusion, tr, te, seed=0, epochs=120):
    trl = DataLoader(tr, batch_size=32, shuffle=True); tel = DataLoader(te, batch_size=32)
    torch.manual_seed(seed); m = DualShieldGNN(hid=48, fusion=fusion)
    opt = torch.optim.Adam(m.parameters(), lr=0.002, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train()
        for b in trl:
            opt.zero_grad(); lf(m(b), b.y).backward(); opt.step()
        sch.step()
    m.eval(); c = t = 0
    with torch.no_grad():
        for b in tel:
            c += (m(b).argmax(-1) == b.y).sum().item(); t += b.y.size(0)
    return c / t

def main():
    data = load()
    print(f"Clean-data 5-fold CV on {len(data)} packages\n")
    out = {}
    for fusion in ["noise_aware", "static", "attention"]:
        accs = [train_eval(fusion, tr, te, seed=i) for i, (tr, te) in enumerate(folds(data))]
        out[fusion] = {"mean": round(float(np.mean(accs)), 3), "std": round(float(np.std(accs)), 3),
                       "folds": [round(a, 3) for a in accs]}
        print(f"  {fusion:12} {out[fusion]['mean']:.3f} +/- {out[fusion]['std']:.3f}")
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results" / "layer3_cleancv.json", "w"), indent=2)
    print("\nSaved results/layer3_cleancv.json")

if __name__ == "__main__":
    main()
