"""Generate paper/sections/cpt_table.tex from results/cpt_analysis.json (no hand-typed numbers)."""
import json
A = json.load(open("results/cpt_analysis.json"))
EXPNAME = {"A": "1\\,GB", "B1": "1\\,GB + warm-up", "B3": "3\\,GB + warm-up"}
ARMN = {"R0": "\\RO{}", "R1": "\\RI{}", "R2": "\\RII{}"}
L = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
     r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lllrrrrrrrr}", r"\toprule",
     r"Model & Data & Arm & Tokens & ne BPB$\downarrow$ & en BPB$\downarrow$ & FLORES ne$\downarrow$ & Belebele ne & chrF ne$\to$en & chrF en$\to$ne & Gen.\ B/s \\",
     r"\midrule"]
f = lambda r, k, fmt: format(r[k], fmt) if r and r.get(k) is not None else "--"
for m, d in A.items():
    b = d.get("base")
    if b:
        L.append(f"{d['name']} & -- & base & 0 & {f(b,'bpb_chunk2k_ne','.3f')} & {f(b,'bpb_chunk2k_en','.3f')} & "
                 f"{f(b,'bpb_flores_ne','.3f')} & {f(b,'belebele_ne','.3f')} & {f(b,'chrf_ne_en','.1f')} & "
                 f"{f(b,'chrf_en_ne','.1f')} & {f(b,'gen_bytes_per_s','.0f')} \\\\")
    for e, x in d["exps"].items():
        ev = x["evals"]
        best_ne = min(r["bpb_chunk2k_ne"] for r in ev.values())
        for a in ("R0", "R1", "R2"):
            r = ev.get(a)
            if not r:
                continue
            ne = f(r, "bpb_chunk2k_ne", ".3f")
            if r["bpb_chunk2k_ne"] == best_ne:
                ne = f"\\textbf{{{ne}}}"
            tok = r.get("train", {}).get("tokens")
            L.append(f" & {EXPNAME[e]} & {ARMN[a]} & {tok/1e6:.0f}M & {ne} & {f(r,'bpb_chunk2k_en','.3f')} & "
                     f"{f(r,'bpb_flores_ne','.3f')} & {f(r,'belebele_ne','.3f')} & {f(r,'chrf_ne_en','.1f')} & "
                     f"{f(r,'chrf_en_ne','.1f')} & {f(r,'gen_bytes_per_s','.0f')} \\\\" if tok else "")
        L.append(r"\addlinespace")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}}",
      r"\caption{Continued pretraining on identical bytes per arm (one seed). BPB: bits per byte on held-out native text split into ${\sim}$2{,}000-byte pieces (primary metric) and on FLORES-200 devtest. Belebele: zero-shot accuracy (900 items, chance 0.25), scored on answer text. chrF++: 5-shot FLORES-200 devtest translation. Gen.\ B/s: UTF-8 bytes of Nepali generated per second (batch 64, greedy). Tokens: training tokens seen (for B1, excluding the embedding warm-up on the first 10\% of the same stream). Bold: best Nepali BPB within each block. All BPB here use the registered prefix; the B1 Llama rows are dominated by the scoring artifact of \S\ref{sec:cpt}, and Table~\ref{tab:prefix} gives the training-separator comparison.}",
      r"\label{tab:cpt}", r"\end{table*}"]
open("paper/sections/cpt_table.tex", "w").write("\n".join(l for l in L if l) + "\n")
tests = {m: {e: x["tests"] for e, x in d["exps"].items()} for m, d in A.items()}
print(json.dumps(tests, indent=1)[:1500])
