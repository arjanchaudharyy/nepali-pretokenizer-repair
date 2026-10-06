import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.spines.top":False,"axes.spines.right":False})
rows=json.load(open("results/audit_unique.json"))
rows=sorted(rows,key=lambda r:r["premium"])
C={"letter-regex":"#c0392b","mark-regex":"#2563eb","no-regex":"#7f8c8d","other-regex":"#d4a017"}
fig,ax=plt.subplots(figsize=(3.3,4.6))
for i,r in enumerate(rows):
    lab=r["name"]+(f" (+{len(r['aliases'])})" if r["aliases"] else "")
    ax.barh(i,r["premium"],color=C[r["kind"]],alpha=0.9 if r["lossless"] else 0.35,height=0.72,
            hatch=None if r["lossless"] else "////",edgecolor="white",linewidth=0)
    ax.errorbar(r["premium"],i,xerr=[[r["premium"]-r["lo"]],[r["hi"]-r["premium"]]],color="k",lw=0.6,capsize=1.2)
    if r["kind"]=="letter-regex" and r["floor"]:
        ax.plot([r["floor"]]*2,[i-0.42,i+0.42],color="k",lw=1.4)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r["name"]+(f" (+{len(r['aliases'])})" if r["aliases"] else "") for r in rows],fontsize=6.6)
ax.axvline(1,color="k",lw=0.5,ls=":")
ax.set_xlabel("Nepali / English token premium")
ax.set_xlim(0,8)
ax.legend(handles=[Patch(color=C["letter-regex"],label=r"letters-only regex ($\backslash$p{L}+)"),
                   Patch(color=C["mark-regex"],label="mark-aware regex"),
                   Patch(color=C["other-regex"],label="other regex (BLOOM)"),
                   Patch(color=C["no-regex"],label="no regex (SentencePiece etc.)"),
                   Line2D([0],[0],color="k",lw=1.4,label="pre-token floor (letters-only)"),
                   Patch(facecolor="#999",alpha=0.35,hatch="////",label="lossy (decode$\\neq$input)")],
          loc="lower right",fontsize=5.8,frameon=False)
fig.tight_layout()
fig.savefig("paper/fig_audit.pdf",bbox_inches="tight"); fig.savefig("figures/fig_audit.png",dpi=200,bbox_inches="tight")
