"""Appendix table: R2's whole Devanagari strings registered as Hugging Face added tokens on the
unchanged regex, against R0/R1/R2. Reads results/revision/added_tokens_v2.json only.
Run from the repo root: .venv/bin/python make_added_tokens_table.py"""
import json
D = json.load(open("results/revision/added_tokens_v2.json"))["models"]
M = ("Llama-3.2-1B", "Qwen3-1.7B-Base")
ROWS = (("R0", r"\RO{}"), ("R1_K32000", r"\RI{}, $K{=}32$k"), ("R2_K32000", r"\RII{}, $K{=}32$k"),
        ("AT2", r"\RII{} strings, added"), ("AT2_sw", r"\quad whole words only"))
num = lambda n: f"{n:,}".replace(",", "{,}")
L = [r"\begin{table}[!htb]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrr}", r"\toprule",
     r" & \multicolumn{2}{c}{Premium} & \multicolumn{2}{c}{Mark-initial} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
     r"Tokenizer & Llama-3 & Qwen3 & Llama-3 & Qwen3 \\", r"\midrule"]
for k, lab in ROWS:
    c = [f"{D[m]['compression'][k]['premium']:.3f}" for m in M]
    mi = [f"{100 * D[m]['mark_initial'][k]['all']:.0f}\\%" for m in M]
    L.append(f"{lab} & " + " & ".join(c + mi) + r" \\")
L.append(r"\midrule")
p = lambda m, k: D[m]["pathologies"]["AT2"][k]
L.append(r"Added-token share & " + " & ".join(f"{100 * p(m, 'added_token_share'):.0f}\\%" for m in M) + r" & & \\")
L.append(r"Starts mid-word & " + " & ".join(f"{100 * p(m, 'added_starting_inside_word'):.0f}\\%" for m in M) + r" & & \\")
L.append(r"Ends mid-word & " + " & ".join(f"{100 * p(m, 'added_ending_inside_word'):.0f}\\%" for m in M) + r" & & \\")
s = D[M[0]]["strings"]
inv = D[M[0]]["invariance_AT2"]["sentences_without_devanagari"]
assert all(D[m]["invariance_AT2"]["changed_without_devanagari"] == 0 and D[m]["invariance_AT2_sw"]["changed_without_devanagari"] == 0 for m in M)
L += [r"\bottomrule", r"\end{tabular}",
      r"\caption{Added tokens on the unchanged letters-only regex, FLORES-200 devtest. We register the "
      + num(s["kept"]) + r" of \RII{}'s " + num(s["candidates"]) + r" new strings that decode to whole Devanagari text "
      r"(with at most one leading space) as added tokens. Premium: Nepali tokens / English tokens. Mark-initial: share of "
      r"Nepali tokens that start with a combining mark. The bottom rows describe default matching, which ignores word boundaries: the share of tokens that are added tokens, and the share of added-token matches that start or end inside a word."
      r" No sentence without a Devanagari character changes its token ids (" + num(inv)
      + r" FLORES-200 sentences, both models, both matching modes).}",
      r"\label{tab:added}", r"\end{table}"]
out = "\n".join(L) + "\n"
open("paper/sections/added_table.tex", "w").write(out)
print(out)
