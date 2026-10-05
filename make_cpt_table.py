"""Generate paper/sections/cpt_table.tex (appendix: every continued-pretraining run) from
results/cpt_analysis.json (registered prefix) and results/cpt_analysis_evals_eos.json
(training separator). No hand-typed numbers. No bold: the warm-up rows under the
registered prefix are dominated by a scoring artifact, so no single 'best' is meaningful.
Run from the repo root: .venv/bin/python make_cpt_table.py
"""
import json
A = json.load(open("results/cpt_analysis.json"))
E = json.load(open("results/cpt_analysis_evals_eos.json"))
# Run labels per arm row (label is split over two rows where it is long).
EXPNAME = {"A": ["main", "", ""], "B1": ["warm-up", "", ""], "BOS": [r"1\,GB, BOS", "in training", ""]}
ARMN = {"R0": "\\RO{}", "R1": "\\RI{}", "R2": "\\RII{}"}
f = lambda r, k, fmt: format(r[k], fmt) if r and r.get(k) is not None else "--"
L = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
     r"\begin{tabular}{lllrrrrrrrrr}", r"\toprule",
     r" & & & & \multicolumn{2}{c}{held-out ne BPB$\downarrow$} & en BPB$\downarrow$ & FLORES & Belebele & \multicolumn{2}{c}{chrF++} & Gen. \\",
     r"\cmidrule(lr){5-6}\cmidrule(lr){10-11}",
     r"Model & Runs & Arm & Tokens & reg. & sep. & reg. & ne BPB & ne acc. & ne$\to$en & en$\to$ne & B/s \\",
     r"\midrule"]
for m, d in A.items():
    b, be = d.get("base"), E[m].get("base")
    sep_base = "n/a" if m == "llama" else f(be, "bpb_chunk2k_ne", ".3f")
    L.append(f"{d['name']} & none & base & 0 & {f(b,'bpb_chunk2k_ne','.3f')} & {sep_base} & {f(b,'bpb_chunk2k_en','.3f')} & "
             f"{f(b,'bpb_flores_ne','.3f')} & {f(b,'belebele_ne','.3f')} & {f(b,'chrf_ne_en','.1f')} & "
             f"{f(b,'chrf_en_ne','.1f')} & {f(b,'gen_bytes_per_s','.0f')} \\\\")
    for e, x in d["exps"].items():
        for i, a in enumerate(("R0", "R1", "R2")):
            r, re_ = x["evals"].get(a), E[m]["exps"].get(e, {}).get("evals", {}).get(a)
            tok = r["train"]["tokens"]
            L.append(f" & {EXPNAME[e][i]} & {ARMN[a]} & {tok/1e6:.0f}M & {f(r,'bpb_chunk2k_ne','.3f')} & {f(re_,'bpb_chunk2k_ne','.3f')} & "
                     f"{f(r,'bpb_chunk2k_en','.3f')} & {f(r,'bpb_flores_ne','.3f')} & {f(r,'belebele_ne','.3f')} & "
                     f"{f(r,'chrf_ne_en','.1f')} & {f(r,'chrf_en_ne','.1f')} & {f(r,'gen_bytes_per_s','.0f')} \\\\")
        L.append(r"\addlinespace")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}",
      r"\caption{Every continued-pretraining run (one run per arm). Main runs: 1\,GB of Nepali and 1\,GB of English, documents separated by the end-of-text token only. Warm-up runs: the same, after training only the embedding matrix on the first 10\% of the stream (Tokens excludes that stage). "
      r"Held-out BPB is scored in pieces of about 2{,}000 bytes, after the registered prefix (reg.: beginning-of-sequence token for Llama) or the training separator (sep.: end-of-text token); for Qwen the two coincide. "
      r"The base Llama model was never trained without its beginning-of-sequence token, so its separator score is not meaningful (n/a). "
      r"All columns other than sep.\ use the registered prefix; for the Llama warm-up runs they are dominated by the scoring artifact of \S\ref{sec:cpt} and should not be compared with the main runs. "
      r"1\,GB, BOS in training: the BOS-consistent rerun, i.e.\ the main runs repeated with every training document wrapped as BOS, document, end-of-text (third amendment); its sep.\ column is scored after the end-of-text token, which these models never saw directly before text. Belebele, translation and generation speed were not run for it (--). "
      r"Belebele: zero-shot accuracy on 900 items (chance 0.25); every value lies between 0.25 and 0.29 with 95\% intervals of about $\pm$0.03, so it cannot separate the arms. chrF++: 5-shot FLORES-200 devtest translation, single greedy pass, no intervals, keeping only the first line of output (empty if the model starts with a newline), so we draw no conclusions from it. "
      r"Gen.\ B/s: UTF-8 bytes of Nepali generated per second, measured on GPUs shared with concurrent training runs; it is noisy (the base and \RO{} Qwen models share tokenizer and architecture yet differ by 41\%) and we do not draw conclusions from it.}",
      r"\label{tab:cpt}", r"\end{table*}"]
open("paper/sections/cpt_table.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
