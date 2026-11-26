"""Draws the paper's Figure 1 (pipeline overview). Usage: python src/figures/fig1_pipeline.py [output.png]"""
import sys
from pathlib import Path
OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / "results" / "figures" / "fig_pipeline.png")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
plt.rcParams.update({"font.family":"Arial"})
BLUE,ORANGE,TEAL,GREY,INK,EDGE="#2b6cb0","#dd6b20","#2c7a7b","#718096","#1a202c","#4a5568"
TINT={BLUE:"#ebf4ff",ORANGE:"#fff4ec",TEAL:"#e6fffa",GREY:"#f7fafc"}
GREEN,RED,AMBER="#2f855a","#c53030","#b7791f"
W,H=5.5,3.55
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
def box(x0,y0,w,h,col,title,body="",tsize=6.4,bsize=5.4,fc=None,num=None,ls="-"):
    ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle="round,pad=0,rounding_size=0.05",fc=fc or TINT[col],ec=col,lw=0.8,ls=ls,zorder=2))
    cy=y0+h/2
    if body:
        ax.text(x0+w/2,y0+h-0.09,title,ha="center",va="top",fontsize=tsize,fontweight="bold",color=INK,zorder=3)
        ax.text(x0+w/2,y0+h-0.27,body,ha="center",va="top",fontsize=bsize,color=EDGE,zorder=3,linespacing=1.25)
    else:
        ax.text(x0+w/2,cy,title,ha="center",va="center",fontsize=tsize,fontweight="bold",color=INK,zorder=3)
    if num is not None:
        ax.add_patch(Circle((x0,y0+h),0.085,fc=INK,ec="white",lw=0.8,zorder=4))
        ax.text(x0,y0+h-0.003,str(num),ha="center",va="center",fontsize=5.8,fontweight="bold",color="white",zorder=5)
def arrow(p,q,col=EDGE,ls="-",lw=0.8):
    ax.annotate("",xy=q,xytext=p,arrowprops=dict(arrowstyle="-|>",color=col,lw=lw,ls=ls,mutation_scale=6,shrinkA=0,shrinkB=0),zorder=1)
def poly(pts,col=EDGE,lw=0.8):
    xs,ys=zip(*pts[:-1]); ax.plot(xs+(pts[-2][0],),ys+(pts[-2][1],),color=col,lw=lw,zorder=1,solid_capstyle="butt")
    arrow(pts[-2],pts[-1],col)
def lane(y0,h,label):
    ax.add_patch(FancyBboxPatch((0.02,y0),W-0.04,h,boxstyle="round,pad=0,rounding_size=0.06",fc="#fbfcfd",ec="#e2e8f0",lw=0.7,zorder=0))
    ax.text(0.1,y0+h-0.06,label,ha="left",va="top",fontsize=5.8,fontweight="bold",color=GREY,zorder=1)
# ---------------- lane A: specification, built once ----------------
yA,hA=2.5,1.0
lane(yA,hA,"SPECIFICATION  ·  built once per regulation")
bh=0.6; by=yA+0.13
box(0.12,by,0.95,bh,GREY,"Article 9 text","Regulation (EU)\n2024/1689")
box(1.27,by,1.05,bh,BLUE,"Requirement extraction","manual gold standard;\nLLM extraction (Qwen2.5-7B)\nevaluated against it",bsize=5.0,tsize=5.9)
box(2.52,by,1.05,bh,BLUE,"SHACL validation","node shapes +\nSHACL-SPARQL rule;\ndefect benchmark",bsize=5.0)
box(3.77,by,1.6,bh,BLUE,"Validated knowledge graph","17 requirements · 34 nodes\ncheck type and semantic flag",num=1)
for a,b in [(1.07,1.27),(2.32,2.52),(3.57,3.77)]: arrow((a,by+bh/2),(b,by+bh/2))
ax.text(1.27+(3.57-1.27)/2,by+bh+0.07,"Layer 1",ha="center",va="bottom",fontsize=6,fontweight="bold",color=BLUE)
# ---------------- lane B: verification, per package ----------------
yB,hB=0.05,2.35
lane(yB,hB,"VERIFICATION  ·  run for every evidence package")
top=yB+hB-0.3
box(0.12,top-0.75,0.95,0.75,GREY,"Evidence package","14 files:\nrisk register,\nmitigation tracker,\ntest results, …",bsize=5.2)
box(1.27,top-0.75,1.05,0.75,GREY,"Parse evidence","measure noise\nf = blank fields\ns = broken links;\nbuild package graph",bsize=5.2,num=2)
arrow((1.07,top-0.375),(1.27,top-0.375))
# stage 3: L2 above L3
x3,w3=2.52,1.5
box(x3,top-0.55,w3,0.55,ORANGE,"Layer 2 · semantic check","hierarchy level · severity\n→ match / mismatch / ambiguous",num=3)
box(x3,top-1.43,w3,0.66,TEAL,"Layer 3 · dual-shield GNN","feature + structure shields,\nnoise-aware fusion (f, s)\n→ class + confidence")
arrow((x3+w3/2,top-0.55),(x3+w3/2,top-0.77),ORANGE)
ax.text(x3+w3/2+0.05,top-0.66,"flags as node features",ha="left",va="center",fontsize=4.9,style="italic",color=ORANGE)
# parse -> L2 and L3
arrow((2.32,top-0.31),(x3,top-0.31))
ax.plot([2.32,2.42],[top-0.31,top-0.31],color=EDGE,lw=0.8)
poly([(2.42,top-0.31),(2.42,top-1.1),(x3,top-1.1)])
# KG -> stage 3 (specification)
poly([(4.2,by),(4.2,yA-0.04),(3.6,yA-0.04),(3.6,top)],col=BLUE)
ax.text(3.53,top+0.13,"requirement specification",ha="right",va="center",fontsize=4.9,style="italic",color=BLUE)
# stage 4: combined verdict
x4,w4=4.22,1.16
y4=top-1.43; h4=0.98
box(x4,y4,w4,h4,GREY,"","",num=4)
ax.text(x4+w4/2,y4+h4-0.09,"Combined verdict",ha="center",va="top",fontsize=6.4,fontweight="bold",color=INK,zorder=3)
rules=[("confidence < 0.75","review",AMBER),("non-compliant","non-compliant",RED),("compliant + L2 flag","review",AMBER),("compliant, no flag","compliant",GREEN)]
for i,(c,o,col) in enumerate(rules):
    yy=y4+h4-0.32-i*0.16
    ax.text(x4+0.07,yy,c,ha="left",va="center",fontsize=4.9,color=EDGE,zorder=3)
    ax.text(x4+w4-0.06,yy,"→ "+o,ha="right",va="center",fontsize=4.9,fontweight="bold",color=col,zorder=3)
# blank title duplicate removed by drawing box with empty title: fix by overdraw
arrow((x3+w3,y4+0.33),(x4,y4+0.33))
poly([(x3+w3,top-0.28),(x3+w3+0.08,top-0.28),(x3+w3+0.08,y4+h4-0.12),(x4,y4+h4-0.12)],col=ORANGE)
# stage 5: report
y5=yB+0.09; h5=0.32
box(0.12,y5,W-0.24-0.0,h5,GREY,"","",num=5)
ax.text(0.27,y5+h5/2,"Audit report",ha="left",va="center",fontsize=6.4,fontweight="bold",color=INK,zorder=3)
ax.text(1.05,y5+h5/2,"verdict (compliant / non-compliant / human review) · confidence · measured noise · semantic flags · references to requirement nodes",ha="left",va="center",fontsize=5.2,color=EDGE,zorder=3)
arrow((x4+w4/2,y4),(x4+w4/2,y5+h5))
fig.savefig(OUT,dpi=600,facecolor="white")
