"""Train the dual-shield GNN. Applies noise on-the-fly for robustness."""
import json, random, shutil, sys
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"dataset"))
from build_graph import build_graph
from model import DualShieldGNN
from inject_noise import inject_noise

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/"data"/"synthetic"/"base"
LABELS = json.load(open(ROOT/"data"/"synthetic"/"labels.json"))

def load_dataset(noise=True, seed=0):
    random.seed(seed)
    data=[]
    tmp = ROOT/"data"/"synthetic"/"_train_tmp"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    for pid, info in LABELS.items():
        src = BASE/pid
        if noise and random.random()<0.6:  # noise 60% of packages
            fl,sl = round(random.uniform(0,0.6),2), round(random.uniform(0,0.5),2)
            dst = tmp/pid; inject_noise(src,dst,fl,sl); src=dst
        g = build_graph(src)
        g.y = torch.tensor([0 if info["label"]=="compliant" else 1])
        data.append(g)
    shutil.rmtree(tmp, ignore_errors=True)
    return data

def run(fusion="noise_aware", epochs=30):
    ds = load_dataset(noise=True)
    random.shuffle(ds)
    n=len(ds); split=int(0.8*n)
    tr, te = ds[:split], ds[split:]
    trl = DataLoader(tr, batch_size=16, shuffle=True)
    tel = DataLoader(te, batch_size=16)

    model = DualShieldGNN(fusion=fusion)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lossf = torch.nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for b in trl:
            opt.zero_grad()
            out = model(b)
            loss = lossf(out, b.y)
            loss.backward(); opt.step()

    # eval
    model.eval(); correct=tot=0
    with torch.no_grad():
        for b in tel:
            pred = model(b).argmax(-1)
            correct += (pred==b.y).sum().item(); tot += b.y.size(0)
    acc = correct/tot
    print(f"  fusion={fusion:12} test accuracy={acc:.3f} (n={tot})")
    return acc

if __name__=="__main__":
    print("Training dual-shield GNN (noise applied on-the-fly)...")
    for fu in ["static","attention","noise_aware"]:
        run(fusion=fu)
