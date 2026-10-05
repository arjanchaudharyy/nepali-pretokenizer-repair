"""Main-text CPT table: held-out Nepali BPB per arm and paired differences with
document-clustered 95% bootstrap intervals.

Reads only results JSON (no hand-typed numbers):
  results/cpt_analysis.json            absolute BPB, registered prefix (BOS for Llama)
  results/cpt_analysis_evals_eos.json  absolute BPB, training separator (EOS)
  results/revision/cluster_bootstrap.json  paired differences, document-clustered intervals
  results/cpt_analysis.json (train.tokens) training tokens per arm
  results/revision/gen_efficiency.json     decoding steps for the same Nepali text

Writes paper/sections/prefix_table.tex.  Run from the repo root:
  .venv/bin/python make_prefix_table_clustered.py
"""
import json

REG = json.load(open("results/cpt_analysis.json"))
SEP = json.load(open("results/cpt_analysis_evals_eos.json"))
CB = json.load(open("results/revision/cluster_bootstrap.json"))
GEN = json.load(open("results/revision/gen_efficiency.json"))

# (column label, absolute-BPB source, model, experiment, cluster-bootstrap prefix key)
COLS = [
    ("llama", "A", REG, "registered_BOS"),
    ("llama", "A", SEP, "training_separator_EOS"),
    ("llama", "B1", SEP, "training_separator_EOS"),
    ("llama", "BOS", REG, "registered_BOS"),  # BOS-consistent rerun (amendment 4): BOS in training
    ("qwen", "A", REG, "registered_BOS"),  # Qwen has no BOS: both prefixes are identical
]
ARMS = (("R0", r"\RO{}"), ("R1", r"\RI{}"), ("R2", r"\RII{}"))
PAIRS = (("R1", "R0"), ("R2", "R0"), ("R2", "R1"))


def bpb(src, m, e, a):
    return src[m]["exps"][e]["evals"][a]["bpb_chunk2k_ne"]


def test(m, e, pfx, x, y):
    return CB["tests"][f"{pfx}|{e}_{m}|ne|{x}-{y}"]


def sign(v):
    return f"{v:+.3f}".replace("-", "$-$")


# The B1 rows under the registered prefix are dominated by the BOS artifact; they are in
# Table tab:cpt (appendix), not here. Check that Qwen's two prefixes really coincide.
for a, _ in ARMS:
    assert abs(bpb(REG, "qwen", "A", a) - bpb(SEP, "qwen", "A", a)) < 1e-9

L = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4.5pt}",
     r"\begin{tabular}{lccccc}", r"\toprule",
     r" & \multicolumn{4}{c}{Llama-3.2-1B} & Qwen3-0.6B \\",
     r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}",
     r" & \multicolumn{2}{c}{main runs} & warm-up runs & BOS rerun & main runs \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}",
     r" & registered prefix & training sep. & training sep. & registered prefix & (both identical) \\",
     r"\midrule",
     r"\multicolumn{6}{l}{\emph{Held-out Nepali BPB} (lower is better)} \\"]
lb = REG["llama"]["base"]["bpb_chunk2k_ne"]
qb = REG["qwen"]["base"]["bpb_chunk2k_ne"]
L.append(f"Base model (untrained) & {lb:.3f} & n/a & n/a & {lb:.3f} & {qb:.3f} \\\\")
for a, lab in ARMS:
    cells = [f"{bpb(src, m, e, a):.3f}" for m, e, src, _ in COLS]
    L.append(f"{lab} & " + " & ".join(cells) + r" \\")
L += [r"\midrule",
      r"\multicolumn{6}{l}{\emph{Paired difference in held-out Nepali BPB} (positive: first arm worse), document-clustered 95\% interval} \\"]
for x, y in PAIRS:
    ts = [test(m, e, pfx, x, y) for m, e, _, pfx in COLS]
    lab = dict(ARMS)[x] + r" $-$ " + dict(ARMS)[y]
    L.append(f"{lab} & " + " & ".join(sign(t["diff"]) for t in ts) + r" \\")
    L.append(" & " + " & ".join(f"[{sign(t['cluster_lo'])}, {sign(t['cluster_hi'])}]" for t in ts) + r" \\[2pt]")
L[-1] = L[-1].replace(r"\\[2pt]", r"\\")
tok = lambda m, e="A": " / ".join(f"{REG[m]['exps'][e]['evals'][a]['train']['tokens'] / 1e6:.0f}" for a, _ in ARMS)
r = GEN["ratios"]
L += [r"\midrule",
      f"Training tokens (M) & \\multicolumn{{3}}{{c}}{{{tok('llama')}}} & {tok('llama', 'BOS')} & {tok('qwen')} \\\\",
      f"Decoding steps & \\multicolumn{{4}}{{c}}{{"
      f"{r['llama']['R0_over_R2']:.2f} / {r['llama']['R1_over_R2']:.2f} / 1}} & "
      f"{r['qwen']['R0_over_R2']:.2f} / {r['qwen']['R1_over_R2']:.2f} / 1 \\\\",
      r"\bottomrule", r"\end{tabular}"]
nd = CB["docs"]["ne"]
L += [r"\caption{Continued pretraining, one run per arm, on the same 1\,GB of Nepali and 1\,GB of English. \textbf{Our results are the BOS-rerun (Llama) and Qwen columns}; the other Llama columns show the scoring artifact of \S\ref{sec:cpt}. "
      r"BPB: held-out native Nepali, in pieces of about 2{,}000 bytes, scored after the registered prefix (BOS for Llama; end-of-text for Qwen, which has no BOS) or the training separator (end-of-text; a post hoc diagnostic). "
      f"Intervals resample {nd['docs']} documents and reflect only the sampling of evaluation text. "
      r"Training tokens and decoding steps (FLORES Nepali, relative to \RII{}) are \RO{} / \RI{} / \RII{}. Secondary metrics: Table~\ref{tab:cpt}.}",
      r"\label{tab:prefix}", r"\end{table*}"]
out = "\n".join(L).replace(f"{nd['pieces']:,}", f"{nd['pieces']:,}".replace(",", "{,}")) + "\n"
open("paper/sections/prefix_table.tex", "w").write(out)
print(out)
