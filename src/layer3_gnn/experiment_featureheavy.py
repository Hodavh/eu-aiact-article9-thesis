"""Firm up the RQ3 headline: feature-heavy noise, 10 seeds per fusion mode."""
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
FEAT=(0.4,0.6); STRUCT=(0.0,0.1)  # feature-heavy

def load(seed):
    random.seed(seed); data=[]
    tmp=ROOT/"data"/"synthetic"/"_fh_tmp"; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir(parents=True)
    for pid,info in LABELS.items():
        fl,sl=round(random.uniform(*FEAT),2),round(random.uniform(*STRUCT),2)
        dst=tmp/pid; inject_noise(BASE/pid,dst,fl,sl)
        g=build_graph(dst); g.y=torch.tensor([0 if info["label"]=="compliant" else 1]); data.append(g)
    shutil.rmtree(tmp,ignore_errors=True); return data

def one_run(fusion, seed, epochs=80):
    ds=load(seed); random.Random(seed).shuffle(ds)
    sp=int(0.8*len(ds)); tr,te=ds[:sp],ds[sp:]
    trl=DataLoader(tr,batch_size=16,shuffle=True); tel=DataLoader(te,batch_size=16)
    torch.manual_seed(seed)
    m=DualShieldGNN(hid=48,fusion=fusion); opt=torch.optim.Adam(m.parameters(),lr=0.008)
    lf=torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train()
        for b in trl: opt.zero_grad(); lf(m(b),b.y).backward(); opt.step()
    m.eval(); c=t=0
    with torch.no_grad():
        for b in tel: c+=(m(b).argmax(-1)==b.y).sum().item(); t+=b.y.size(0)
    return c/t

def main():
    print("RQ3 headline: feature-heavy noise, 10 seeds each\n")
    out={}
    for fu in ["static","attention","noise_aware"]:
        accs=[one_run(fu,s) for s in range(10)]
        out[fu]={"mean":round(float(np.mean(accs)),3),"std":round(float(np.std(accs)),3),"runs":accs}
        print(f"  {fu:12} {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    json.dump(out, open(ROOT/"results"/"layer3_featureheavy.json","w"), indent=2)
    print("\nSaved results/layer3_featureheavy.json")

if __name__=="__main__": main()
