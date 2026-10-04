import regex
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fp=fm.FontProperties(fname="/System/Library/Fonts/Kohinoor.ttc")
mono=fm.FontProperties(family="monospace")
PAT_A=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
PAT_B=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
s="नेपालको सरकार"
rows=[("letters-only \\p{L}+",[p.strip() for p in regex.findall(PAT_A,s) if p.strip()],"#c0392b"),
      ("repaired [\\p{L}\\p{M}]+",[p.strip() for p in regex.findall(PAT_B,s) if p.strip()],"#2563eb")]
fig,ax=plt.subplots(figsize=(3.3,0.85)); ax.axis("off")
for r,(lab,toks,c) in enumerate(rows):
    y=0.72-r*0.5
    ax.text(0,y,lab,fontsize=6.5,va="center",fontproperties=mono)
    x=0.60
    for t in toks:
        w=0.026*len(t)+0.035
        ax.add_patch(plt.Rectangle((x,y-0.17),w,0.34,fc=c,alpha=0.12,ec=c,lw=0.6))
        ax.text(x+w/2,y,t if not regex.match(r"\p{M}",t) else "◌"+t,fontproperties=fp,fontsize=8,ha="center",va="center")
        x+=w+0.01
ax.set_xlim(0,1.22); ax.set_ylim(-0.1,1)
fig.savefig("paper/fig_split.pdf",bbox_inches="tight",pad_inches=0.02); fig.savefig("figures/fig_split.png",dpi=250,bbox_inches="tight")
print(rows)
