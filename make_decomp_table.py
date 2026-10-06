"""Premium decomposition (appendix Table tab:decomp) for every letters-only and mark-aware tokenizer.

premium = base x regex factor x vocabulary factor, on FLORES-200 devtest:
  base   = |P_rep(ne)| / |T(en)|   (P_rep: the repaired Llama-3 pattern, the same for every tokenizer)
  regex  = |P(ne)| / |P_rep(ne)|   (P: the tokenizer's FULL pre-tokenizer pipeline, not only its last regex)
  vocab  = |T(ne)| / |P(ne)|
The floor in the audit table is base x regex = |P(ne)| / |T(en)|.

Writes results/decomp.json and paper/sections/decomp_table.tex. No hand-typed numbers.
Run from the repo root: .venv/bin/python make_decomp_table.py   (downloads tokenizer files only)
"""
import json, unicodedata
import regex
import tiktoken
from transformers import AutoTokenizer

PAT_REP = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+|\p{N}{1,3}"
           r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
REP = regex.compile(PAT_REP)
L = lambda c: [unicodedata.normalize("NFC", x) for x in open(f"flores200_dataset/devtest/{c}.devtest").read().splitlines()]
NE, EN = L("npi_Deva"), L("eng_Latn")
P_REP = sum(len(REP.findall(s)) for s in NE)

# (label in table, aliases shown as "(+n)", source)  -- source: ("hf", repo) or ("tt", tiktoken encoding)
LETTERS = [("GPT-2", 0, ("tt", "gpt2")), ("LFM2", 0, ("hf", "LiquidAI/LFM2-8B-A1B")),
           ("Falcon3", 0, ("hf", "tiiuae/Falcon3-7B-Base")), (r"\texttt{cl100k}", 3, ("tt", "cl100k_base")),
           ("GLM-4.5", 0, ("hf", "zai-org/GLM-4.5")), ("Qwen2.5", 1, ("hf", "Qwen/Qwen2.5-7B")),
           ("GLM-5.3", 0, ("hf", "zai-org/GLM-5.3")), ("Llama-3", 1, ("hf", "NousResearch/Meta-Llama-3-8B"))]
MARKS = [("DeepSeek-V3", 1, ("hf", "deepseek-ai/DeepSeek-V3")), ("TituLLM", 0, ("hf", "hishab/titulm-llama-3.2-1b-v2.0")),
         ("Qwen3.5", 0, ("hf", "Qwen/Qwen3.5-9B")), ("Mistral-NeMo", 1, ("hf", "unsloth/Mistral-Nemo-Base-2407")),
         ("Llama-4", 0, ("hf", "unsloth/Llama-4-Scout-17B-16E-Instruct")), (r"\texttt{o200k}", 1, ("tt", "o200k_base")),
         ("Arkios", 0, ("hf", "sajalregmi4/arkios-tokenizer"))]


def measure(src):
    kind, name = src
    if kind == "tt":
        e = tiktoken.get_encoding(name)
        pat = regex.compile(e._pat_str)
        enc = lambda s: e.encode(s)
        pre = lambda s: pat.findall(s)
    else:
        t = AutoTokenizer.from_pretrained(name)
        enc = lambda s: t(s, add_special_tokens=False)["input_ids"]
        pre = lambda s: t.backend_tokenizer.pre_tokenizer.pre_tokenize_str(s)
    Tn, Te = sum(len(enc(s)) for s in NE), sum(len(enc(s)) for s in EN)
    P = sum(len(pre(s)) for s in NE)
    return dict(premium=Tn / Te, base=P_REP / Te, regex=P / P_REP, vocab=Tn / P, floor=P / Te)


out, rows = {}, {}
for block, items in (("letters", LETTERS), ("marks", MARKS)):
    rows[block] = []
    for label, al, src in items:
        r = measure(src)
        out[label] = dict(r, source=src[1], block=block)
        rows[block].append((label, al, r))
        print(f"{label:<16} premium={r['premium']:.2f} base={r['base']:.2f} regex={r['regex']:.2f} vocab={r['vocab']:.2f} floor={r['floor']:.3f}")
json.dump(out, open("results/decomp.json", "w"), indent=1)

T = [r"\begin{table}[!htb]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrr}", r"\toprule",
     r"Tokenizer & Premium & Base & Regex & Vocab. \\", r"\midrule"]
for block in ("letters", "marks"):
    for label, al, r in sorted(rows[block], key=lambda x: -x[2]["premium"]):
        name = label + (f" (+{al})" if al else "")
        T.append(f"{name} & {r['premium']:.2f} & {r['base']:.2f} & {r['regex']:.2f} & {r['vocab']:.2f} \\\\")
    T.append(r"\midrule")
T[-1] = r"\bottomrule"
T += [r"\end{tabular}",
      r"\caption{Premium = base $\times$ regex factor $\times$ vocabulary factor, on FLORES-200 devtest, for all 15 letters-only and mark-aware tokenizers in the audit (top: letters-only; bottom: mark-aware; ``(+$n$)'' counts aliases). Merge-based extension can only lower the last column, and not below 1.}",
      r"\label{tab:decomp}", r"\end{table}"]
open("paper/sections/decomp_table.tex", "w").write("\n".join(T) + "\n")
print("\n".join(T))
