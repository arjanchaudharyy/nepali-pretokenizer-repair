import json, glob, os
P={c:json.load(open(f"results/prep.{c}.json")) for c in ["npi_Deva","eng_Latn"]}
M={os.path.basename(os.path.dirname(f)):json.load(open(f)) for f in glob.glob("results/pretok/*/meta.json")}
L=[r"\begin{table}[t]",r"\centering\small",r"\setlength{\tabcolsep}{3.5pt}",r"\begin{tabular}{lrr}",r"\toprule",
   r"& Nepali & English \\",r"\midrule"]
for k,lab in [("docs","documents read"),("short","dropped: under 200 chars"),("dup","dropped: exact duplicate"),("contam","dropped: FLORES 10-gram"),("kept","documents kept")]:
    L.append(f"{lab} & {P['npi_Deva'][k]:,} & {P['eng_Latn'][k]:,} \\\\")
L.append(f"held-out (MB) & {P['npi_Deva']['held']/1e6:.0f} & {P['eng_Latn']['held']/1e6:.0f} \\\\")
L.append(r"\midrule")
L.append(r"\multicolumn{3}{l}{CPT tokens per arm (millions), 3\,GB + 3\,GB}\\")
for m,lab in [("llama","Llama-3.2-1B"),("qwen","Qwen3-0.6B")]:
    for a in ["R0","R1","R2"]:
        x=M.get(f"{m}_{a}")
        if x: L.append(f"{lab} {a} & {x['ne_tokens']/1e6:,.0f} & {x['en_tokens']/1e6:,.0f} \\\\")
L+=[r"\bottomrule",r"\end{tabular}",r"\caption{Corpus cleaning (FineWeb-2 \texttt{npi\_Deva}; FineWeb-Edu) and the token counts of each arm's identical 6\,GB training stream. English token counts differ across arms of a model by less than 0.003\%, from the few English documents that contain Devanagari or combining marks.}",r"\label{tab:data}",r"\end{table}"]
open("paper/sections/data_table.tex","w").write("\n".join(L)+"\n"); print("\n".join(L))
