"""Generate paper/sections/extra_runs_table.tex: every run of amendments 5-7 (paper: fourth to sixth
amendments) next to the seed-0 run it extends. No hand-typed numbers.
Run from the repo root: .venv/bin/python make_extra_runs_table.py
"""
import json
from pathlib import Path
EV, RU = Path("results/cpt/evals"), Path("results/cpt/runs")
ROWS = [  # (model, arm label, run name, lr, seed, steps note)
    ("Llama", r"\RO{}", "bos_llama_R0_s0", "1e-4", 0), ("Llama", r"\RO{}", "bos_llama_R0_s1", "1e-4", 1),
    ("Llama", r"\RI{}", "bos_llama_R1_s0", "1e-4", 0), ("Llama", r"\RI{}", "bos_llama_R1_s1", "1e-4", 1),
    ("Llama", r"\RII{}", "bos_llama_R2_s0", "1e-4", 0), ("Llama", r"\RII{}", "bos_llama_R2_s1", "1e-4", 1),
    ("Llama", r"\RII{}-m", "ms_bos_llama_R2_s0", "1e-4", 0), ("Llama", r"\RII{}-m", "ms_bos_llama_R2_s1", "1e-4", 1),
    ("Qwen", r"\RO{}", "qwen_R0_s0", "1e-4", 0), ("Qwen", r"\RO{}", "qwen_R0_s1", "1e-4", 1),
    ("Qwen", r"\RI{}", "qwen_R1_s0", "1e-4", 0), ("Qwen", r"\RI{}", "qwen_R1_s1", "1e-4", 1),
    ("Qwen", r"\RII{}", "qwen_R2_s0", "1e-4", 0), ("Qwen", r"\RII{}", "qwen_R2_s1", "1e-4", 1),
    ("Qwen", r"\RII{}-m", "ms_qwen_R2_s0", "1e-4", 0), ("Qwen", r"\RII{}-m", "ms_qwen_R2_s1", "1e-4", 1),
    ("Qwen", r"\RO{}", "lr3_qwen_R0_s0", "3e-4", 0), ("Qwen", r"\RI{}", "lr3_qwen_R1_s0", "3e-4", 0),
    ("Qwen", r"\RII{}-m", "lr3_ms_qwen_R2_s0", "3e-4", 0),
]
L = [r"\begin{table}[!htb]", r"\centering\small", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{lllrrrr}", r"\toprule",
     r"Model & Arm & LR & Seed & Steps & ne BPB & en BPB \\", r"\midrule"]
prev = None
for m, arm, run, lr, seed in ROWS:
    e = json.load(open(EV / f"{run}.json"))
    steps = json.loads(open(RU / run / "train_log.jsonl").read().strip().splitlines()[-1])["steps"]
    if prev and prev != m:
        L.append(r"\midrule")
    prev = m
    lrs = {"1e-4": r"$10^{-4}$", "3e-4": r"$3{\times}10^{-4}$"}[lr]
    L.append(f"{m} & {arm} & {lrs} & {seed} & {steps} & {e['bpb_chunk2k_ne']:.4f} & {e['bpb_chunk2k_en']:.4f} \\\\")
L += [r"\bottomrule", r"\end{tabular}",
      r"\caption{Every continued-pretraining run of the Llama BOS setting and of Qwen, including the matched-steps, second-seed and learning-rate runs of the fourth to seventh amendments. BPB: held-out text in pieces of about 2{,}000 bytes after the registered prefix. \RII{}-m: \RII{} trained for \RI{}'s step count, with 35\% of its stream seen twice. Seeds change data order (and bf16 non-determinism) only.}",
      r"\label{tab:extra-runs}", r"\end{table}"]
open("paper/sections/extra_runs_table.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
