"""Non-redundant graph builder: compliance signal SPLIT across channels.
Structural-only signals (R6 mitigation links) live in edges.
Feature-only signals (R9 acceptability, severity) live in node features.
NO summary node, so neither channel alone can recover the full verdict."""
import csv, json
from pathlib import Path
import torch
from torch_geometric.data import Data

SEV={"negligible":1,"minor":2,"serious":3,"critical":4,"catastrophic":5,"":0}
ACC={"acceptable":1.0,"not_acceptable":0.0,"":0.5}

def _safe_float(v,d=0.0):
    try: return float(v)
    except: return d
def _rc(p): return list(csv.DictReader(open(p))) if p.exists() else []

def estimate_noise(pkg):
    d=Path(pkg); tf=bf=tr=br=0
    rids={r["risk_id"] for r in _rc(d/"clinical_risk_register.csv")}
    for fn in ["clinical_risk_register.csv","mitigation_tracker.csv"]:
        for row in _rc(d/fn):
            for k,v in row.items():
                if k.endswith("_id"): continue
                tf+=1
                if v=="" or v is None: bf+=1
    for m in _rc(d/"mitigation_tracker.csv"):
        tr+=1
        if m.get("linked_risk_id") not in rids: br+=1
    return (round(bf/tf,3) if tf else 0.0, round(br/tr,3) if tr else 0.0)

# node features (feature-channel signal ONLY): [is_risk, is_mit, completeness, severity, acceptability]
# NO mitigation-link info in features; NO summary node.
# R6 (all risks mitigated) is recoverable ONLY from edges.
# R9 (acceptability) is recoverable ONLY from features.
def build_graph(pkg, semantic_flags=None):
    d=Path(pkg)
    risks=_rc(d/"clinical_risk_register.csv"); mits=_rc(d/"mitigation_tracker.csv")
    nodes, idx=[],{}
    for r in risks:
        idx[("risk",r["risk_id"])]=len(nodes)
        comp=sum(1 for v in r.values() if v!="")/max(len(r),1)
        nodes.append([1,0,comp,SEV.get(r.get("severity",""),0)/5, 0.0])
    for m in mits:
        idx[("mit",m["mitigation_id"])]=len(nodes)
        comp=sum(1 for v in m.values() if v!="")/max(len(m),1)
        acc=ACC.get(m.get("acceptability",""),0.5)
        nodes.append([0,1,comp,0, acc])   # acceptability = feature-only signal (R9)

    # edges = mitigation->risk links = structure-only signal (R6)
    edges=[[],[]]
    for m in mits:
        rk=("risk",m.get("linked_risk_id"))
        if rk in idx:
            a,b=idx[("mit",m["mitigation_id"])],idx[rk]
            edges[0]+=[a,b]; edges[1]+=[b,a]

    x=torch.tensor(nodes,dtype=torch.float) if nodes else torch.zeros((1,5))
    ei=torch.tensor(edges,dtype=torch.long) if edges[0] else torch.zeros((2,0),dtype=torch.long)
    f,s=estimate_noise(pkg)
    data=Data(x=x,edge_index=ei); data.noise=torch.tensor([f,s],dtype=torch.float)
    return data

if __name__=="__main__":
    import sys
    g=build_graph(sys.argv[1] if len(sys.argv)>1 else "data/synthetic/base/PKG_C0000")
    print(f"nodes={g.x.shape[0]} feats={g.x.shape[1]} edges={g.edge_index.shape[1]} noise={g.noise.tolist()}")
