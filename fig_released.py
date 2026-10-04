import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.spines.top":False,"axes.spines.right":False})
rows=json.load(open("results/released_extensions.json"))["rows"]
keep=[r for r in rows if r.get("repair_headroom") and r["repo"].startswith(("atsuki","LiquidAI")) and "amh" not in r.get("lang","") ]
lab=[]
for r in keep:
    b=r.get("base","")
    base=("LFM2.5" if "LFM2" in b else "Qwen3" if "Qwen3" in b else "Qwen2.5" if "Qwen2.5" in b else "Llama-3.1" if "3.1" in b else "Llama-3")
    LANG={"tel":"Telugu","sin":"Sinhala","mya":"Burmese","tam":"Tamil","ben":"Bengali","guj":"Gujarati","hin":"Hindi","npi":"Nepali","tha":"Thai"}
    lab.append(f"{base}, {LANG[r.get('lang','')[:3]]}")
fig,ax=plt.subplots(figsize=(3.3,3.2))
for i,r in enumerate(keep):
    b=r["base_over_floor"]; e=r["ext_over_floor"]; rf=1/r["repair_headroom"]
    ax.plot([rf,b],[i,i],color="#ddd",lw=1,zorder=0)
    ax.scatter(b,i,marker="x",color="#555",s=14,zorder=2)
    ax.scatter(e,i,color="#c0392b",s=16,zorder=3)
    ax.scatter(rf,i,marker="|",color="#2563eb",s=60,lw=1.6,zorder=3)
ax.axvline(1,color="#c0392b",ls="--",lw=0.8)
ax.set_xscale("log")
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
ax.xaxis.set_major_locator(FixedLocator([0.2,0.5,1,2,4,8])); ax.xaxis.set_major_formatter(FixedFormatter(["0.2","0.5","1","2","4","8"]))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_yticks(range(len(keep))); ax.set_yticklabels(lab,fontsize=6.5); ax.invert_yaxis()
ax.set_xlabel("tokens ÷ letters-only pre-token floor (log)")
ax.scatter([],[],marker="x",color="#555",s=14,label="base tokenizer")
ax.scatter([],[],color="#c0392b",s=16,label="released extension")
ax.scatter([],[],marker="|",color="#2563eb",s=60,lw=1.6,label="floor after repair")
ax.legend(fontsize=6,frameon=False,loc="lower center",bbox_to_anchor=(0.45,1.0),ncol=3,handletextpad=0.2,columnspacing=0.8)
fig.tight_layout()
fig.savefig("paper/fig_released.pdf",bbox_inches="tight"); fig.savefig("figures/fig_released.png",dpi=220,bbox_inches="tight")
print(len(keep), lab)
