"""Table: Nepali BPB and paired differences under the registered prefix and the training separator."""
import json
A = json.load(open("results/cpt_analysis.json"))
E = json.load(open("results/cpt_analysis_evals_eos.json"))
NAME = {"A": "A", "B1": "B1"}
ARMN = {"R0": "\\RO{}", "R1": "\\RI{}", "R2": "\\RII{}"}


def d(t):
    return f"${t["diff"]:+.3f}$ $[{t["lo"]:+.3f}, {t["hi"]:+.3f}]$" if t else "--"


L = [r"\begin{table}[t]", r"\centering", r"\setlength{\tabcolsep}{3pt}",
     r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llrr}", r"\toprule",
     r"Model, exp. & Comparison & Registered prefix & Training separator \\", r"\midrule"]
for m in ("llama", "qwen"):
    for e in ("A", "B1"):
        a, b = A[m]["exps"].get(e), E[m]["exps"].get(e)
        if not a or not b:
            continue
        first = True
        for k, lab in (("ne:R1-R0", "\\RI{} $-$ \\RO{}"), ("ne:R2-R0", "\\RII{} $-$ \\RO{}"), ("ne:R2-R1", "\\RII{} $-$ \\RI{}")):
            ta, tb = a["tests"].get(k), b["tests"].get(k)
            if not ta and not tb:
                continue
            head = f"{A[m]['name'].split('-')[0]}, {NAME[e]}" if first else ""
            first = False
            ra = d(ta) + ("$^\\dagger$" if e == "B1" and ta else "")
            L.append(f"{head} & {lab} & {ra} & {d(tb)} \\\\")
        L.append(r"\addlinespace")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}}",
      r"\caption{Paired differences in held-out Nepali bits per byte (positive: first arm worse), 95\% bootstrap intervals. A: 1\,GB + 1\,GB; B1: the same with an embedding warm-up. The Qwen \RII{} $-$ \RI{} upper bound is $+0.0101$, just above the 0.01 margin. Registered prefix: the model's beginning-of-sequence token where it has one (Llama), as pre-registered; Qwen has none and uses its end-of-text token in both columns. Training separator: the end-of-text token that preceded every document during continued pretraining. $^\dagger$Dominated by the scoring artifact of \S\ref{sec:cpt}: the warm-up shifts the beginning-of-sequence embedding.}",
      r"\label{tab:prefix}", r"\end{table}"]
open("paper/sections/prefix_table.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
