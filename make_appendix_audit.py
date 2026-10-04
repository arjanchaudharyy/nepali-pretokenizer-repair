import json
rows=json.load(open("results/audit_unique.json"))
sw={r["name"]:r for r in json.load(open("results/sweep_devtest.json"))["rows"]}
K={"letter-regex":"letters","mark-regex":"mark-aware","no-regex":"none","other-regex":"other"}
L=[r"\begin{table*}[t]",r"\centering\small",r"\setlength{\tabcolsep}{4pt}",
   r"\begin{tabular}{lllrrrrrl}",r"\toprule",
   r"Tokenizer & Aliases & Regex & $|V|$ & ne/en & 95\% CI & Floor & Mark-init. & Lossless \\",r"\midrule"]
for r in sorted(rows,key=lambda r:r["premium"]):
    al=", ".join(r["aliases"]) if r["aliases"] else "--"
    fl=f"{r['floor']:.2f}" if r["floor"] is not None else "--"
    L.append(f"{r['name']} & {al} & {K[r['kind']]} & {r['vocab']:,} & {r['premium']:.2f} & [{r['lo']:.2f}, {r['hi']:.2f}] & {fl} & {100*r['mark_initial']:.0f}\\% & {'yes' if r['lossless'] else 'no'} \\\\".replace(",",",\\allowbreak ") if False else
             f"{r['name']} & {al} & {K[r['kind']]} & {r['vocab']:,} & {r['premium']:.2f} & [{r['lo']:.2f}, {r['hi']:.2f}] & {fl} & {100*r['mark_initial']:.0f}\\% & {'yes' if r['lossless'] else 'no'} \\\\")
L+= [r"\bottomrule",r"\end{tabular}",
     r"\caption{All 24 distinct tokenizers (33 released names) on FLORES-200 devtest. ne/en: Nepali/English token premium with 95\% paired-bootstrap interval. Floor: lowest premium reachable by vocabulary extension under the tokenizer's own regex (letters-only and mark-aware regex tokenizers only). Mark-init.: share of Nepali tokens whose first character is a combining mark. Lossless: decoding reproduces the input on 200 Nepali sentences. Hugging Face repositories are listed in the released code.}",
     r"\label{tab:audit-full}",r"\end{table*}"]
open("paper/sections/audit_table.tex","w").write("\n".join(L)+"\n")
print(len(rows))
