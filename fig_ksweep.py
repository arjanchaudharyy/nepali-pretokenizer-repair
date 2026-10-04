import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.spines.top":False,"axes.spines.right":False})
rows=[r for r in json.load(open("results/ksweep.json")) if r["split"]=="devtest"]
fig,axs=plt.subplots(1,2,figsize=(3.3,1.9),sharey=True)
for ax,m,title in zip(axs,["Llama-3.2-1B","Qwen3-1.7B-Base"],["Llama-3","Qwen3"]):
    r0=[r for r in rows if r["model"]==m and r["arm"]=="R0"][0]["premium"]
    for arm,c,lab in [("R1","#c0392b","R1 extend"),("R2","#2563eb","R2 repair+extend")]:
        rs=sorted([r for r in rows if r["model"]==m and r["arm"]==arm],key=lambda r:r["K"])
        K=[r["K"]/1000 for r in rs]; P=[r["premium"] for r in rs]
        ax.plot(K,P,"o-",color=c,ms=2.5,lw=1.1,label=lab)
        ax.fill_between(K,[r["lo"] for r in rs],[r["hi"] for r in rs],color=c,alpha=0.15,lw=0)
        ax.axhline(rs[0]["floor_premium"],color=c,ls="--",lw=0.7)
    ax.axhline(r0,color="k",lw=0.6); ax.text(1.05,r0+0.07,f"original {r0:.2f}",fontsize=6)
    ax.axhline(1.61,color="#555",ls=":",lw=0.7); ax.text(13,1.68,"o200k 1.61",fontsize=6,color="#555")
    ax.set_xscale("log",base=2); ax.set_xticks([1,4,16,64]); ax.set_xticklabels(["1k","4k","16k","64k"])
    ax.set_title(title,fontsize=8); ax.set_xlabel("new merges $K$")
    ax.set_ylim(0.6,4.7)
axs[0].set_ylabel("Nepali / English premium")
axs[1].legend(fontsize=6,frameon=False,loc="upper right",bbox_to_anchor=(1.0,0.86))
fig.tight_layout(pad=0.3)
fig.savefig("paper/fig_ksweep.pdf",bbox_inches="tight"); fig.savefig("figures/fig_ksweep.png",dpi=220,bbox_inches="tight")
