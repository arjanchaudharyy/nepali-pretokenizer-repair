"""
ksweep_eval.py: compression for every retrofit tokenizer in tok/<model>/<arm>_K<k>.

For each: Nepali premium vs English on FLORES dev and devtest (paired bootstrap
CI), English token ids identical to base (bool), lossless (bool), and the
pre-token floor of its regex. Writes results/ksweep.json.
"""
import json, re, sys, unicodedata
from pathlib import Path
import numpy as np, regex
from tokenizers import Tokenizer
from transformers import AutoTokenizer

H = Path("/home/ntt")
FL = H / "flores200_dataset"
BASES = {"Llama-3.2-1B": "unsloth/Llama-3.2-1B", "Qwen3-1.7B-Base": "Qwen/Qwen3-1.7B-Base"}


def load(code, split):
    return [unicodedata.normalize("NFC", x) for x in (FL / split / f"{code}.{split}").read_text().splitlines()]


def boot(a, b, B=2000):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    r = a[idx].sum(1) / b[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def counts(tk, sents):
    return np.array([len(e.ids) for e in tk.encode_batch(sents, add_special_tokens=False)])


rows = []
for short, repo in BASES.items():
    base = AutoTokenizer.from_pretrained(repo).backend_tokenizer
    for split in ("dev", "devtest"):
        ne, en = load("npi_Deva", split), load("eng_Latn", split)
        b_ne, b_en = counts(base, ne), counts(base, en)
        lo, hi = boot(b_ne, b_en)
        rows.append(dict(model=short, arm="R0", K=0, split=split, premium=b_ne.sum() / b_en.sum(), lo=lo, hi=hi,
                         en_identical=True, lossless=True))
        for d in sorted((H / "tok" / short).glob("R*_K*")):
            if not (d / "tokenizer.json").exists():
                continue
            arm, K = d.name.split("_K")
            tk = Tokenizer.from_file(str(d / "tokenizer.json"))
            c_ne, c_en = counts(tk, ne), counts(tk, en)
            lo, hi = boot(c_ne, c_en)
            meta = json.load(open(d / "meta.json"))
            floor = sum(len(regex.findall(meta["regex"], s)) for s in ne)
            same = all(base.encode(s, add_special_tokens=False).ids == tk.encode(s, add_special_tokens=False).ids
                       for s in en)
            lossless = all(tk.decode(tk.encode(s, add_special_tokens=False).ids) == s for s in ne[:300] + en[:300])
            rows.append(dict(model=short, arm=arm, K=int(K), split=split, premium=float(c_ne.sum() / c_en.sum()),
                             lo=lo, hi=hi, floor_premium=floor / float(b_en.sum()), en_identical=same,
                             lossless=lossless, new_tokens=meta["new_tokens"]))
            print(short, split, arm, K, round(rows[-1]["premium"], 3), same, lossless, flush=True)
Path(H / "results").mkdir(exist_ok=True)
json.dump(rows, open(H / "results/ksweep.json", "w"), indent=1)
