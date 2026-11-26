"""Draws the paper's Figure 2 (Article 9 requirement graph). Usage: python src/figures/fig2_article9_graph.py [output.png]"""
import sys
from pathlib import Path
OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / "results" / "figures" / "fig_article9_kg.png")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Arc
plt.rcParams.update({"font.family":"Arial"})
BLUE,ORANGE,GREY,LGREY,INK,EDGE="#2b6cb0","#dd6b20","#718096","#cbd5e0","#1a202c","#4a5568"
R=[("R1",1,""),("R2",2,""),("R3",2,"(a)"),("R4",2,"(b)"),("R5",2,"(c)"),("R6",2,"(d)"),("R7",3,""),("R8",4,""),
   ("R9",5,""),("R10",5,"(a)"),("R11",5,"(b)"),("R12",5,"(c)"),("R13",6,""),("R14",7,""),("R15",8,""),("R16",9,""),("R17",10,"")]
COL={"R4":ORANGE,"R10":ORANGE,"R11":ORANGE,"R12":ORANGE,"R7":GREY,"R17":LGREY}
EDGES=[("R6","R3"),("R8","R6"),("R9","R6"),("R10","R2"),("R10","R6"),("R11","R6"),("R12","R6"),("R13","R6"),
       ("R15","R13"),("R16","R1"),("R16","R13"),("R16","R2")]
EXT=[("R5","Article 72","post-market\nmonitoring"),("R12","Article 13","information to\ndeployers"),("R14","Article 60","real-world\ntesting")]
W,H=5.5,2.75
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
u=0.262; g=1.42*u; x0=0.2
x={};cur=x0;prev=None
for rid,p,_ in R:
    if prev is not None: cur+= u if p==prev else g
    x[rid]=cur; prev=p
r=0.105; Y=1.55
for s,t in EDGES:
    a,b=x[s],x[t]; w=abs(a-b); h=0.10+0.20*w
    ax.add_patch(Arc(((a+b)/2,Y+r*0.7),w,2*h,theta1=0,theta2=180,color=EDGE,lw=0.7,zorder=1))
    ax.annotate("",xy=(b,Y+r+0.005),xytext=(b,Y+r+0.09),arrowprops=dict(arrowstyle="-|>",color=EDGE,lw=0.7,mutation_scale=5,shrinkA=0,shrinkB=0),zorder=2)
for rid,p,sub in R:
    c=COL.get(rid,BLUE)
    ax.add_patch(Circle((x[rid],Y),r,fc=c,ec=GREY if c==LGREY else c,lw=0.6,zorder=3))
    ax.text(x[rid],Y-0.004,rid,ha="center",va="center",color=INK if c==LGREY else "white",fontsize=5.6,fontweight="bold",zorder=4)
    if sub: ax.text(x[rid],Y-r-0.05,sub,ha="center",va="top",fontsize=5.5,color=EDGE,bbox=dict(fc="white",ec="none",pad=0.6),zorder=5)
paras={}
for rid,p,_ in R: paras.setdefault(p,[]).append(x[rid])
yb=Y-r-0.24
for p,xs in paras.items():
    lo,hi=min(xs)-0.1,max(xs)+0.1
    ax.plot([lo,lo,hi,hi],[yb+0.035,yb,yb,yb+0.035],color=GREY,lw=0.6)
    ax.text((lo+hi)/2,yb-0.04,f"¶{p}",ha="center",va="top",fontsize=6.5,fontweight="bold",color=INK,bbox=dict(fc="white",ec="none",pad=0.6),zorder=5)
ax.text(x["R10"],yb-0.2,"mitigation hierarchy (levels 1–3)",ha="center",va="top",fontsize=5.3,style="italic",color=ORANGE,bbox=dict(fc="white",ec="none",pad=0.6),zorder=5)
ye=yb-0.62
for src,name,desc in EXT:
    xx=x[src]; bw=0.6
    ax.add_patch(FancyBboxPatch((xx-bw/2,ye-0.19),bw,0.38,boxstyle="round,pad=0,rounding_size=0.04",fc="white",ec=GREY,lw=0.6,ls=(0,(2.5,1.5)),zorder=3))
    ax.text(xx,ye+0.09,name,ha="center",va="center",fontsize=5.8,fontweight="bold",color=INK,zorder=4)
    ax.text(xx,ye-0.065,desc,ha="center",va="center",fontsize=5.0,color=EDGE,zorder=4,linespacing=1.05)
    ax.annotate("",xy=(xx,ye+0.195),xytext=(xx,Y-r),arrowprops=dict(arrowstyle="-|>",color=GREY,lw=0.6,ls=(0,(2.5,1.5)),mutation_scale=5,shrinkA=0,shrinkB=0),zorder=2)
# (ye+0.195 is the box top)
# legend: two rows
ly=0.22
def dot(px,py,c,l):
    ax.add_patch(Circle((px,py),0.045,fc=c,ec=GREY if c==LGREY else c,lw=0.5)); ax.text(px+0.08,py,l,va="center",fontsize=5.6,color=INK)
def arr(px,py,c,ls,l):
    ax.annotate("",xy=(px+0.22,py),xytext=(px-0.04,py),arrowprops=dict(arrowstyle="-|>",color=c,lw=0.6,ls=ls,mutation_scale=5,shrinkA=0,shrinkB=0)); ax.text(px+0.28,py,l,va="center",fontsize=5.6,color=INK)
c1,c2,c3=0.24,2.05,3.75
dot(c1,ly,BLUE,"Structural check"); dot(c2,ly,ORANGE,"Semantic check (Layer 2)"); arr(c3,ly,EDGE,"-","Cross-reference within Article 9")
dot(c1,ly-0.17,GREY,"Structural constraint (not scored)"); dot(c2,ly-0.17,LGREY,"Permissive provision (not scored)"); arr(c3,ly-0.17,GREY,(0,(2.5,1.5)),"Reference to another Article")
fig.savefig(OUT,dpi=600,facecolor="white")
