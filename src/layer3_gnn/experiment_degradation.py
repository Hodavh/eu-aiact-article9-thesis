"""RQ4: degradation curve. Accuracy vs increasing noise; identify usable-reliability threshold."""
import json, random, shutil, sys
from pathlib import Path
import numpy as np, torch
from torch_geometric.loader import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"dataset"))
from build_graph import build_graph
from model import DualShieldGNN
from structured_noise import apply_structured_noise

ROOT=Path(__file__).resolve().parents[2]
DIFF=ROOT/"data"/"synthetic"/"diffuse"
LABELS=json.load(open(ROOT/"data"/"synthetic"/"labels_diffuse.json"))

# graduated noise levels (feature=structure, symmetric increase)
LEVELS=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
THRESHOLD=0.70  # usable-reliability line

def build_all(level,seed=0):
    random.seed(seed)
    tmp=ROOT/"data"/"synthetic"/"_deg"; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir(parents=True)
    data=[]
    for pid,info in LABELS.items():
        dst=tmp/pid; apply_structured_noise(DIFF/pid,dst,level,level)
        g=build_graph(dst); g.y=torch.tensor([0 if info["label"]=="compliant" else 1]); data.append(g)
    shutil.rmtree(tmp,ignore_errors=True); random.Random(123).shuffle(data); return data

def folds(data,k=5):
    sz=len(data)//k
    for i in range(k): yield data[:i*sz]+data[(i+1)*sz:], data[i*sz:(i+1)*sz]

def train_eval(tr,te,seed=0,epochs=120):
    trl=DataLoader(tr,batch_size=32,shuffle=True); tel=DataLoader(te,batch_size=32)
    torch.manual_seed(seed); m=DualShieldGNN(hid=48,fusion="noise_aware")
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
    curve={}; threshold_crossed=None
    print(f"RQ4: degradation curve (usable threshold = {THRESHOLD})\n")
    print(f"{'noise':>7} {'accuracy':>12}")
    for lv in LEVELS:
        data=build_all(lv)
        accs=[train_eval(tr,te,seed=i) for i,(tr,te) in enumerate(folds(data))]
        mean=float(np.mean(accs)); std=float(np.std(accs))
        curve[lv]={"mean":round(mean,3),"std":round(std,3)}
        mark=" <-- below usable" if mean<THRESHOLD and threshold_crossed is None else ""
        if mean<THRESHOLD and threshold_crossed is None: threshold_crossed=lv
        print(f"{lv:>7.1f} {mean:.3f}±{std:.2f}{mark}")
    out={"threshold":THRESHOLD,"first_level_below_threshold":threshold_crossed,"curve":curve}
    json.dump(out, open(ROOT/"results"/"layer3_degradation.json","w"), indent=2)
    print(f"\nUsable-reliability threshold ({THRESHOLD}) first crossed at noise level: {threshold_crossed}")
    print("Saved results/layer3_degradation.json")

if __name__=="__main__": main()
