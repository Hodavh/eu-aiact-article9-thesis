"""RQ3 on the DIFFUSE dataset: attention vs noise-aware across asymmetry.
Diffuse violations require aggregating signal across nodes + both channels."""
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

CONDITIONS={"symmetric_30_30":(0.30,0.30),"asym_20_10":(0.20,0.10),
            "asym_50_10":(0.50,0.10),"asym_70_10":(0.70,0.10)}

def build_all(fl,sl,seed=0):
    random.seed(seed)
    tmp=ROOT/"data"/"synthetic"/"_diffsw"; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir(parents=True)
    data=[]
    for pid,info in LABELS.items():
        dst=tmp/pid; apply_structured_noise(DIFF/pid,dst,fl,sl)
        g=build_graph(dst); g.y=torch.tensor([0 if info["label"]=="compliant" else 1]); data.append(g)
    shutil.rmtree(tmp,ignore_errors=True); random.Random(123).shuffle(data); return data

def folds(data,k=5):
    sz=len(data)//k
    for i in range(k): yield data[:i*sz]+data[(i+1)*sz:], data[i*sz:(i+1)*sz]

def train_eval(fusion,tr,te,seed=0,epochs=120):
    trl=DataLoader(tr,batch_size=32,shuffle=True); tel=DataLoader(te,batch_size=32)
    torch.manual_seed(seed); m=DualShieldGNN(hid=48,fusion=fusion)
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
    out={}
    print(f"{'condition':16} {'static':>10} {'attention':>12} {'noise_aware':>12}")
    for name,(fl,sl) in CONDITIONS.items():
        data=build_all(fl,sl); fds=list(folds(data))
        st=[train_eval("static",tr,te,seed=i) for i,(tr,te) in enumerate(fds)]
        at=[train_eval("attention",tr,te,seed=i) for i,(tr,te) in enumerate(fds)]
        na=[train_eval("noise_aware",tr,te,seed=i) for i,(tr,te) in enumerate(fds)]
        out[name]={"static":round(float(np.mean(st)),3),"attention":round(float(np.mean(at)),3),
                   "noise_aware":round(float(np.mean(na)),3),
                   "static_std":round(float(np.std(st)),3),"attention_std":round(float(np.std(at)),3),
                   "noise_aware_std":round(float(np.std(na)),3)}
        print(f"{name:16} {np.mean(st):.3f}±{np.std(st):.2f} {np.mean(at):.3f}±{np.std(at):.2f} {np.mean(na):.3f}±{np.std(na):.2f}")
    json.dump(out, open(ROOT/"results"/"layer3_structured.json","w"), indent=2)
    print("\nSaved results/layer3_diffuse.json")

if __name__=="__main__": main()
