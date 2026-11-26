"""Layer 3: package -> graph + noise estimate. Encodes all violation signals."""
import csv, json
from pathlib import Path
import torch
from torch_geometric.data import Data

CSV_FILES = ["clinical_risk_register.csv","mitigation_tracker.csv",
             "clinical_validation_study.csv","traceability_matrix.csv"]
SEV = {"negligible":1,"minor":2,"serious":3,"critical":4,"catastrophic":5,"":0}
ACC = {"acceptable":1.0,"not_acceptable":0.0,"":0.5}

def _read_csv(p): return list(csv.DictReader(open(p))) if p.exists() else []
def _read_json(p):
    try: return json.load(open(p))
    except: return []

def _safe_float(v, default=0.0):
    try: return float(v)
    except: return default

def estimate_noise(pkg):
    d=Path(pkg); tf=bf=tr=br=0
    rids={r["risk_id"] for r in _read_csv(d/"clinical_risk_register.csv")}
    for fn in CSV_FILES:
        for row in _read_csv(d/fn):
            for v in row.values():
                tf+=1
                if v=="" or v is None: bf+=1
    for m in _read_csv(d/"mitigation_tracker.csv"):
        tr+=1
        if m.get("linked_risk_id") not in rids: br+=1
    return (round(bf/tf,3) if tf else 0.0, round(br/tr,3) if tr else 0.0)

# node feature layout (9 dims):
# [is_risk, is_mit, is_summary, completeness, severity_ord, hierarchy_lvl,
#  acceptability, semantic_flag, package_signal]
def build_graph(pkg, semantic_flags=None):
    d=Path(pkg)
    risks=_read_csv(d/"clinical_risk_register.csv")
    mits=_read_csv(d/"mitigation_tracker.csv")
    nodes, idx = [], {}

    for r in risks:
        idx[("risk",r["risk_id"])]=len(nodes)
        comp=sum(1 for v in r.values() if v!="")/max(len(r),1)
        nodes.append([1,0,0, comp, SEV.get(r.get("severity",""),0)/5, 0, 0, 0, 0])
    for m in mits:
        idx[("mit",m["mitigation_id"])]=len(nodes)
        comp=sum(1 for v in m.values() if v!="")/max(len(m),1)
        lvl=_safe_float(m.get("hierarchy_level"))/3
        acc=ACC.get(m.get("acceptability",""),0.5)
        sflag=0
        if semantic_flags:
            f=next((s for s in semantic_flags if s.get("record_id")==m["mitigation_id"]),None)
            if f and f.get("flag")=="MISMATCH": sflag=float(f.get("severity",1))/3
        nodes.append([0,1,0, comp, 0, lvl, acc, sflag, 0])

    # package-level summary node: carries signals from non-graph files
    # rms_established (R1), test pass-rate (R15), post-market present (R5), risks-mitigated (R6)
    sysdoc=_read_json(d/"system_documentation.pdf.txt")
    rms=1.0 if (sysdoc and str(sysdoc[0].get("rms_established","")).lower()=="true") else 0.0
    res=_read_json(d/"validation_test_results.json")
    passrate=sum(1 for x in res if x.get("pass_fail")=="pass")/len(res) if res else 0.0
    pm=1.0 if (d/"post_market_surveillance.csv").exists() or (d/"model_drift_log.json").exists() else 0.0
    rids={r["risk_id"] for r in risks}
    linked={m.get("linked_risk_id") for m in mits}
    mitrate=len(rids & linked)/len(rids) if rids else 0.0
    # pack 4 signals into the summary node's feature slots
    summary=[0,0,1, mitrate, 0, 0, 0, 0, (rms+passrate+pm)/3]
    sidx=len(nodes); nodes.append(summary)

    edges=[[],[]]
    for m in mits:
        rk=("risk",m.get("linked_risk_id"))
        if rk in idx:
            a,b=idx[("mit",m["mitigation_id"])],idx[rk]
            edges[0]+=[a,b]; edges[1]+=[b,a]
    # connect summary node to all others (so its signal propagates)
    for i in range(sidx):
        edges[0]+=[sidx,i]; edges[1]+=[i,sidx]

    x=torch.tensor(nodes,dtype=torch.float) if nodes else torch.zeros((1,9))
    ei=torch.tensor(edges,dtype=torch.long) if edges[0] else torch.zeros((2,0),dtype=torch.long)
    feat,struct=estimate_noise(pkg)
    data=Data(x=x, edge_index=ei); data.noise=torch.tensor([feat,struct],dtype=torch.float)
    return data

if __name__=="__main__":
    import sys
    pkg=sys.argv[1] if len(sys.argv)>1 else "data/synthetic/base/PKG_C0000"
    g=build_graph(pkg)
    print(f"{pkg}: nodes={g.x.shape[0]} feats={g.x.shape[1]} edges={g.edge_index.shape[1]} noise={g.noise.tolist()}")
