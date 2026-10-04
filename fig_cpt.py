"""Figure: held-out Nepali and English bits per byte vs training compute, per arm."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
A = json.load(open("results/cpt_analysis.json"))
C = {"R0": "#555555", "R1": "#c0392b", "R2": "#2563eb"}
LAB = {"R0": "R0 no extension", "R1": "R1 extend", "R2": "R2 repair+extend"}
fig, axs = plt.subplots(2, 2, figsize=(3.3, 3.0), sharex="col")
for col, m in enumerate(["llama", "qwen"]):
    d = A.get(m)
    if not d:
        continue
    for row, lang in enumerate(["ne", "en"]):
        ax = axs[row, col]
        base = d["evals"].get("base", {}).get(f"bpb_chunk2k_{lang}")
        if base:
            ax.axhline(base, color="#999", ls=":", lw=0.8)
        for arm in ["R0", "R1", "R2"]:
            r = d["evals"].get(f"{arm}_s0")
            if not r or "train" not in r:
                continue
            tok = r["train"]["tokens"] / 1e6
            ax.scatter(tok, r[f"bpb_chunk2k_{lang}"], color=C[arm], s=18, zorder=3, label=LAB[arm])
            eq = d["eqcompute"].get(arm)
            r2 = d["evals"].get("R2_s0")
            if eq and r2 and "train" in r2:
                t2 = r2["train"]["tokens"] / 1e6
                ax.scatter(t2, eq[f"bpb_chunk2k_{lang}"], facecolor="white", edgecolor=C[arm], s=18, zorder=3)
                ax.plot([t2, tok], [eq[f"bpb_chunk2k_{lang}"], r[f"bpb_chunk2k_{lang}"]], color=C[arm], lw=0.6)
        if row == 0:
            ax.set_title(d["name"], fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{'Nepali' if lang == 'ne' else 'English'} BPB")
        if row == 1:
            ax.set_xlabel("training tokens (M)")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=6, frameon=False)
fig.tight_layout(pad=0.3)
fig.savefig("paper/fig_cpt.pdf", bbox_inches="tight")
fig.savefig("figures/fig_cpt.png", dpi=220, bbox_inches="tight")
