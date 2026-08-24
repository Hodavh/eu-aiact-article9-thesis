"""Fusion comparison on the NON-REDUNDANT dataset under asymmetric noise.
Hypothesis: because each violation kind is visible in only ONE channel,
asymmetric noise should make smart (noise-aware) fusion matter.
Honest: report whatever happens - tie or win."""
import json, random, shutil, sys
from pathlib import Path
import numpy as np, torch
from torch_geometric.loader import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"dataset"))
from build_graph_nonredundant import build_graph
from model import DualShieldGNN
from structured_noise import apply_structured_noise

ROOT=Path(__file__).resolve().parents[2]
NR=ROOT/"data"/"synthetic"/"nonredundant"
LABELS=json.load(open(ROOT/"data"/"synthetic"/"labels_nonredundant.json"))

CONDITIONS={
    "clean":(0.0,0.0),
    "feature_heavy":(0.5,0.1),   # features corrupted -> must lean structural
    "structure_heavy":(0.1,0.5), # structure corrupted -> must lean feature
    "symmetric":(0.3,0.3),
}

def build_all(fn,sn,seed=0):
    random.seed(seed)
    tmp=ROOT/"data"/"synthetic"/"_nr"; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir(parents=True)
    data=[]
    for pid,info in LABELS.items():
        dst=tmp/pid; apply_structured_noise(NR/pid,dst,fn,sn)
        g=build_graph(dst); g.y=torch.tensor([0 if info["label"]=="compliant" else 1]); data.append(g)
    shutil.rmtree(tmp,ignore_errors=True); random.Random(1).shuffle(data); return data

def folds(data,k=5):
    sz=len(data)//k
    for i in range(k): yield data[:i*sz]+data[(i+1)*sz:], data[i*sz:(i+1)*sz]

def train_eval(tr,te,fusion,seed=0,epochs=120):
    trl=DataLoader(tr,batch_size=32,shuffle=True); tel=DataLoader(te,batch_size=32)
    torch.manual_seed(seed); m=DualShieldGNN(hid=48,fusion=fusion,in_dim=5)
    opt=torch.optim.Adam(m.parameters(),lr=0.002,weight_decay=5e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    lf=torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train()
        for b in trl: opt.zero_grad(); lf(m(b),b.y).backward(); opt.step()
        sch.step()
    m.eval(); c=t=0
    with torch.no_grad():
        for b in tel: c+=(m(b).argmax(-1)==b.y).sum().item(); t+=b.y.size(0)
    return c/t

def main():
    print("Non-redundant fusion comparison (5-fold CV)\n")
    print(f"{'condition':<16}{'static':>9}{'attention':>11}{'noise_aware':>13}{'learned_noise':>15}")
    results={}
    for cond,(fn,sn) in CONDITIONS.items():
        data=build_all(fn,sn)
        row={}
        for fusion in ["static","attention","noise_aware","learned_noise"]:
            accs=[train_eval(tr,te,fusion,seed=i) for i,(tr,te) in enumerate(folds(data))]
            row[fusion]=float(np.mean(accs)); row[fusion+"_std"]=float(np.std(accs))
        results[cond]=row
        print(f"{cond:<16}{row['static']:>9.3f}{row['attention']:>11.3f}{row['noise_aware']:>13.3f}{row['learned_noise']:>15.3f}")
    json.dump(results, open(ROOT/"results"/"layer3_nonredundant.json","w"), indent=2)
    print("\nSaved results/layer3_nonredundant.json")
    # honest verdict
    print("\n--- Honest check: does noise_aware win anywhere? ---")
    for cond,row in results.items():
        ln,att=row["learned_noise"],row["attention"]
        diff=ln-att; sig="YES" if diff>row["learned_noise_std"] else "no (within std)"
        print(f"{cond:<16} learned_noise - attention = {diff:+.3f}  beats attention: {sig}")

if __name__=="__main__": main()
