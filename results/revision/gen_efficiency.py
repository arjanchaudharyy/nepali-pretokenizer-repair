"""
gen_efficiency.py (review R2 W5, R1 W6d): deterministic generation-efficiency ratios.

Replaces measured bytes/s with tokens per byte: decoding one token is one forward step,
so the number of decoding steps needed to emit a fixed text is tokens(text). The ratios
tokens(R0)/tokens(R2) and tokens(R1)/tokens(R2) on FLORES-200 devtest Nepali
(1,012 sentences, NFC) are the deterministic step-count speed-ups. They ignore the
per-step cost difference from the larger LM head of R1/R2 (vocab 160,256 vs 128,256 for
Llama; 183,669 vs 151,669 for Qwen).

Tokenizers: R0 = HF base tokenizer (unsloth/Llama-3.2-1B; Qwen/Qwen3-0.6B-Base, the CPT
model, checked byte-identical in vocab+merges+pre-tokenizer to Qwen/Qwen3-1.7B-Base on
which R1/R2 were built); R1/R2 = tok/box/<model>/R{1,2}_K32000/tokenizer.json (the K=32000
tokenizers copied from the box; no rebuild was needed).
Also lists every measured gen_* field in results/cpt/evals{,_eos}/*.json and eval.py's FLORES
ne_tokens_per_byte (computed by box/eval.py on FLORES Nepali, not on held-out text), to document the inconsistency.

Run from the repo root:  .venv/bin/python results/revision/gen_efficiency.py
Output: results/revision/gen_efficiency.json
"""
import hashlib
import json
import unicodedata
from pathlib import Path
from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
FL = ROOT / "flores200_dataset/devtest"
MODELS = {"llama": ("unsloth/Llama-3.2-1B", "Llama-3.2-1B"), "qwen": ("Qwen/Qwen3-0.6B-Base", "Qwen3-1.7B-Base")}


def load(code):
    return [unicodedata.normalize("NFC", x) for x in (FL / f"{code}.devtest").read_text().splitlines()]


def canon(tk):
    c = json.loads(tk.to_str())
    return hashlib.sha256(json.dumps([c["model"]["vocab"], c["model"]["merges"], c["pre_tokenizer"],
                                      c["normalizer"]], sort_keys=True).encode()).hexdigest()


ne, en = load("npi_Deva"), load("eng_Latn")
nb_ne, nb_en = sum(len(s.encode()) for s in ne), sum(len(s.encode()) for s in en)
out = dict(flores_bytes=dict(ne=nb_ne, en=nb_en), tokenizers={}, ratios={}, measured={})
q06 = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base").backend_tokenizer
q17 = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base").backend_tokenizer
out["qwen_0.6B_vs_1.7B_tokenizer_identical"] = canon(q06) == canon(q17)
for m, (repo, short) in MODELS.items():
    toks = {"R0": AutoTokenizer.from_pretrained(repo).backend_tokenizer}
    for a in ("R1", "R2"):
        toks[a] = Tokenizer.from_file(str(ROOT / "tok/box" / short / f"{a}_K32000" / "tokenizer.json"))
    cnt = {}
    for a, tk in toks.items():
        n_ne = sum(len(e.ids) for e in tk.encode_batch(ne, add_special_tokens=False))
        n_en = sum(len(e.ids) for e in tk.encode_batch(en, add_special_tokens=False))
        cnt[a] = n_ne
        out["tokenizers"][f"{m}_{a}"] = dict(vocab=tk.get_vocab_size(), ne_tokens=n_ne, en_tokens=n_en,
                                             ne_tokens_per_byte=n_ne / nb_ne, ne_bytes_per_token=nb_ne / n_ne,
                                             en_tokens_per_byte=n_en / nb_en)
    out["ratios"][m] = dict(R0_over_R2=cnt["R0"] / cnt["R2"], R1_over_R2=cnt["R1"] / cnt["R2"],
                            R0_over_R1=cnt["R0"] / cnt["R1"])

for evdir in ("evals", "evals_eos"):
    for p in sorted((ROOT / "results/cpt" / evdir).glob("*.json")):
        if p.name.endswith(".per.json"):
            continue
        r = json.load(open(p))
        row = {k: r[k] for k in ("gen_tokens_per_s", "gen_bytes_per_s", "gen_bytes_per_token",
                                 "ne_tokens_per_byte") if k in r}
        if row:
            if "ne_tokens_per_byte" in row:
                row["evalpy_flores_ne_bytes_per_token"] = 1 / row["ne_tokens_per_byte"]
            out["measured"][f"{evdir}/{p.stem}"] = row

json.dump(out, open(ROOT / "results/revision/gen_efficiency.json", "w"), indent=1)
print("qwen 0.6B == 1.7B tokenizer:", out["qwen_0.6B_vs_1.7B_tokenizer_identical"])
for k, v in out["tokenizers"].items():
    print(f"{k:10} V={v['vocab']:>7} ne_tok={v['ne_tokens']:>7} tok/B={v['ne_tokens_per_byte']:.4f} B/tok={v['ne_bytes_per_token']:.2f} en_tok={v['en_tokens']}")
for m, r in out["ratios"].items():
    print(m, {k: round(v, 3) for k, v in r.items()})
for k, v in out["measured"].items():
    print(f"{k:32}", {kk: round(vv, 4) for kk, vv in v.items()})
