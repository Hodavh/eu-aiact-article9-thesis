"""Dual-shield GNN with three fusion modes: static, attention, noise-aware."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ShieldChannel(nn.Module):
    """One GCN channel."""
    def __init__(self, in_dim, hid):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, hid)
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        x = F.relu(self.c2(x, ei))
        return x

class DualShieldGNN(nn.Module):
    """Two channels (feature + structural shield), fused, then classified.
    fusion in {'static','attention','noise_aware'}."""
    def __init__(self, in_dim=9, hid=32, fusion="noise_aware"):
        super().__init__()
        self.fusion = fusion
        self.feat_shield = ShieldChannel(in_dim, hid)
        self.struct_shield = ShieldChannel(in_dim, hid)
        if fusion == "attention":
            self.att = nn.Linear(hid*2, 2)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 2))

    def forward(self, data):
        x, ei = data.x, data.edge_index
        batch = data.batch if hasattr(data,"batch") and data.batch is not None \
                else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        hf = self.feat_shield(x, ei)      # feature channel
        hs = self.struct_shield(x, ei)    # structural channel

        if self.fusion == "static":
            w = torch.tensor([0.5,0.5], device=x.device)
            h = w[0]*hf + w[1]*hs
        elif self.fusion == "attention":
            a = torch.softmax(self.att(torch.cat([hf,hs],dim=-1)), dim=-1)
            h = a[:,0:1]*hf + a[:,1:2]*hs
        else:  # noise_aware: weight by MEASURED noise (down-weight the noisier channel)
            noise = data.noise.view(-1,2)[0] if hasattr(data,"noise") else torch.tensor([0.,0.],device=x.device)
            fn, sn = noise[0], noise[1]
            # channel reliability = 1 - its noise; feature channel trusts features, struct trusts structure
            wf = (1-fn); ws = (1-sn)
            tot = wf+ws+1e-6
            h = (wf/tot)*hf + (ws/tot)*hs

        hg = global_mean_pool(h, batch)
        return self.head(hg)

if __name__=="__main__":
    import sys; sys.path.insert(0,"src/layer3_gnn")
    from build_graph import build_graph
    g = build_graph("data/synthetic/base/PKG_C0000")
    for fu in ["static","attention","noise_aware"]:
        m = DualShieldGNN(fusion=fu)
        out = m(g)
        print(f"  {fu:12} -> logits {out.detach().tolist()}, pred {out.argmax(-1).item()}")
