"""RQ3 (correct framing): fusion modes under symmetric vs asymmetric noise."""
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

# noise regimes: (feature_range, structural_range)
REGIMES={
    "symmetric":     ((0.2,0.4),(0.2,0.4)),
    "feature_heavy": ((0.4,0.6),(0.0,0.1)),
    "struct_heavy":  ((0.0,0.1),(0.4,0.6)),
}

def load(seed, regime):
    random.seed(seed); data=[]
    (flo,fhi),(slo,shi)=REGIMES[regime]
    tmp=ROOT/"data"/"synthetic"/"_asym_tmp"; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir(parents=True)
    for pid,info in LABELS.items():
        fl,sl=round(random.uniform(flo,fhi),2),round(random.uniform(slo,shi),2)
        dst=tmp/pid; inject_noise(BASE/pid,dst,fl,sl)
        g=build_graph(dst); g.y=torch.tensor([0 if info["label"]=="compliant" else 1]); data.append(g)
    shutil.rmtree(tmp,ignore_errors=True); return data

def one_run(fusion, seed, regime, epochs=80):
    ds=load(seed,regime); random.Random(seed).shuffle(ds)
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
    out={}
    for reg in REGIMES:
        print(f"\n=== {reg} noise ===")
        out[reg]={}
        for fu in ["static","attention","noise_aware"]:
            accs=[one_run(fu,s,reg) for s in range(4)]
            out[reg][fu]={"mean":round(float(np.mean(accs)),3),"std":round(float(np.std(accs)),3)}
            print(f"  {fu:12} {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    json.dump(out, open(ROOT/"results"/"layer3_asymmetric.json","w"), indent=2)
    print("\nSaved results/layer3_asymmetric.json")

if __name__=="__main__": main()
