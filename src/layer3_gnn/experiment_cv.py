"""RQ3 with 5-fold cross-validation: fair, low-variance fusion comparison.
Each fusion mode is evaluated on identical folds under each noise regime."""
import json, random, shutil, sys
from pathlib import Path
import numpy as np, torch
from torch_geometric.loader import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"dataset"))
from build_graph import build_graph
from model import DualShieldGNN
from inject_noise import inject_noise

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data"/"synthetic"/"base"
LABELS=json.load(open(ROOT/"data"/"synthetic"/"labels.json"))
REGIMES={"symmetric":((0.2,0.4),(0.2,0.4)),
         "feature_heavy":((0.4,0.6),(0.0,0.1)),
         "struct_heavy":((0.0,0.1),(0.4,0.6))}
K=5

def build_all(regime, seed=0):
    """Build the full noised dataset ONCE per regime (same graphs for all methods)."""
    random.seed(seed)
    (flo,fhi),(slo,shi)=REGIMES[regime]
    tmp=ROOT/"data"/"synthetic"/"_cv_tmp"; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir(parents=True)
    data=[]
    for pid,info in LABELS.items():
        fl,sl=round(random.uniform(flo,fhi),2),round(random.uniform(slo,shi),2)
        dst=tmp/pid; inject_noise(BASE/pid,dst,fl,sl)
        g=build_graph(dst); g.y=torch.tensor([0 if info["label"]=="compliant" else 1]); data.append(g)
    shutil.rmtree(tmp,ignore_errors=True)
    random.Random(123).shuffle(data)  # fixed shuffle -> identical folds for all methods
    return data

def folds(data,k):
    n=len(data); size=n//k
    for i in range(k):
        te=data[i*size:(i+1)*size]
        tr=data[:i*size]+data[(i+1)*size:]
        yield tr,te

def train_eval(fusion, tr, te, epochs=120, seed=0):
    trl=DataLoader(tr,batch_size=32,shuffle=True); tel=DataLoader(te,batch_size=32)
    torch.manual_seed(seed)
    m=DualShieldGNN(hid=48,fusion=fusion); opt=torch.optim.Adam(m.parameters(),lr=0.002,weight_decay=5e-4)
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
    out={}
    for reg in REGIMES:
        print(f"\n=== {reg} noise (5-fold CV) ===")
        data=build_all(reg)               # SAME data for all 3 methods
        fold_list=list(folds(data,K))
        out[reg]={}
        for fu in ["static","attention","noise_aware"]:
            accs=[train_eval(fu,tr,te,seed=i) for i,(tr,te) in enumerate(fold_list)]
            out[reg][fu]={"mean":round(float(np.mean(accs)),3),"std":round(float(np.std(accs)),3)}
            print(f"  {fu:12} {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    json.dump(out, open(ROOT/"results"/"layer3_cv.json","w"), indent=2)
    print("\nSaved results/layer3_cv.json")

if __name__=="__main__": main()
