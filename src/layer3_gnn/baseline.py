"""Single-channel GCN baseline vs dual-shield, under noise. 5-fold CV, stabilised."""
import json, random, shutil, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"dataset"))
from build_graph import build_graph
from model import DualShieldGNN
from inject_noise import inject_noise

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data"/"synthetic"/"base"
LABELS=json.load(open(ROOT/"data"/"synthetic"/"labels.json"))

class SingleGCN(nn.Module):
    """Plain single-channel GCN baseline."""
    def __init__(self, in_dim=9, hid=48):
        super().__init__()
        self.c1=GCNConv(in_dim,hid); self.c2=GCNConv(hid,hid)
        self.head=nn.Sequential(nn.Linear(hid,hid),nn.ReLU(),nn.Linear(hid,2))
    def forward(self,data):
        x,ei=data.x,data.edge_index
        batch=data.batch if hasattr(data,"batch") and data.batch is not None else torch.zeros(x.size(0),dtype=torch.long)
        x=F.relu(self.c1(x,ei)); x=F.relu(self.c2(x,ei))
        return self.head(global_mean_pool(x,batch))

def build_all(seed=0):
    random.seed(seed)
    tmp=ROOT/"data"/"synthetic"/"_bl_tmp"; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir(parents=True)
    data=[]
    for pid,info in LABELS.items():
        fl,sl=round(random.uniform(0.2,0.5),2),round(random.uniform(0.2,0.5),2)  # mixed noise
        dst=tmp/pid; inject_noise(BASE/pid,dst,fl,sl)
        g=build_graph(dst); g.y=torch.tensor([0 if info["label"]=="compliant" else 1]); data.append(g)
    shutil.rmtree(tmp,ignore_errors=True); random.Random(123).shuffle(data); return data

def folds(data,k=5):
    size=len(data)//k
    for i in range(k):
        yield data[:i*size]+data[(i+1)*size:], data[i*size:(i+1)*size]

def train_eval(model_fn, tr, te, seed=0, epochs=120):
    trl=DataLoader(tr,batch_size=32,shuffle=True); tel=DataLoader(te,batch_size=32)
    torch.manual_seed(seed); m=model_fn()
    opt=torch.optim.Adam(m.parameters(),lr=0.002,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    lf=torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train()
        for b in trl: opt.zero_grad(); lf(m(b),b.y).backward(); opt.step()
        sched.step()
    m.eval(); c=t=0
    with torch.no_grad():
        for b in tel: c+=(m(b).argmax(-1)==b.y).sum().item(); t+=b.y.size(0)
    return c/t

def main():
    data=build_all()
    fl=list(folds(data))
    print("Single-channel GCN vs Dual-shield (noise-aware), 5-fold CV, mixed noise\n")
    single=[train_eval(lambda:SingleGCN(), tr,te,seed=i) for i,(tr,te) in enumerate(fl)]
    dual=[train_eval(lambda:DualShieldGNN(hid=48,fusion="noise_aware"), tr,te,seed=i) for i,(tr,te) in enumerate(fl)]
    print(f"  Single-channel GCN:   {np.mean(single):.3f} +/- {np.std(single):.3f}")
    print(f"  Dual-shield (noise):  {np.mean(dual):.3f} +/- {np.std(dual):.3f}")
    json.dump({"single_channel":{"mean":round(float(np.mean(single)),3),"std":round(float(np.std(single)),3)},
               "dual_shield":{"mean":round(float(np.mean(dual)),3),"std":round(float(np.std(dual)),3)}},
              open(ROOT/"results"/"layer3_baseline.json","w"), indent=2)
    print("\nSaved results/layer3_baseline.json")

if __name__=="__main__": main()
