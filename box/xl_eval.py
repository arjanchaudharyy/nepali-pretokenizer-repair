"""
xl_eval.py: compression eval for tok/xl/<code>/<model>/<arm>_K<k> on FLORES devtest.

Per (language, base model): R0 (base) premium; per arm/K: language/English token
premium with paired bootstrap 95% CI (B=2000, seed 0, resampling sentence pairs),
pre-token floor premium under the arm's regex (pre-token count of the language
devtest / base English tokens, as in ksweep_eval.py), lossless round trip on all
1012 language + 1012 English sentences, and English token ids identical to the
base on all 1012 English devtest sentences. Also mark density of the language.

usage: python box/xl_eval.py <code> <model_short>   -> results/xl/<code>.<model>.json
       python box/xl_eval.py --merge                -> results/xling_ksweep.json
"""
import json, sys, unicodedata
from pathlib import Path
import numpy as np, regex
from tokenizers import Tokenizer
from transformers import AutoTokenizer

H = Path(__file__).resolve().parent.parent
FL = H / "flores200_dataset"
BASES = {"Llama-3.2-1B": "unsloth/Llama-3.2-1B", "Qwen3-1.7B-Base": "Qwen/Qwen3-1.7B-Base"}


def load(code, split="devtest"):
    return [unicodedata.normalize("NFC", x) for x in (FL / split / f"{code}.{split}").read_text().splitlines()]


def boot(a, b, B=2000):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    r = a[idx].sum(1) / b[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def counts(tk, sents):
    return np.array([len(e.ids) for e in tk.encode_batch(sents, add_special_tokens=False)])


def mark_density(sents):
    t = "".join(sents)
    return len(regex.findall(r"\p{M}", t)) / len(t)


def run(code, short):
    base = AutoTokenizer.from_pretrained(BASES[short]).backend_tokenizer
    xs, en = load(code), load("eng_Latn")
    b_x, b_en = counts(base, xs), counts(base, en)
    E = float(b_en.sum())
    base_en_ids = [e.ids for e in base.encode_batch(en, add_special_tokens=False)]
    lo, hi = boot(b_x, b_en)
    rows = [dict(lang=code, model=short, arm="R0", K=0, premium=float(b_x.sum() / E), lo=lo, hi=hi,
                 lang_tokens=int(b_x.sum()), en_tokens=int(E), en_identical=True, lossless=True,
                 mark_density=mark_density(xs))]
    for d in sorted((H / "tok/xl" / code / short).glob("R*_K*"), key=lambda p: (p.name[:2], int(p.name.split("_K")[1]))):
        if not (d / "tokenizer.json").exists():
            continue
        arm, K = d.name.split("_K")
        tk = Tokenizer.from_file(str(d / "tokenizer.json"))
        meta = json.load(open(d / "meta.json"))
        c_x, c_en = counts(tk, xs), counts(tk, en)
        lo, hi = boot(c_x, c_en)
        floor = sum(len(regex.findall(meta["regex"], s)) for s in xs)
        same = [e.ids for e in tk.encode_batch(en, add_special_tokens=False)] == base_en_ids
        lossless = all(tk.decode(tk.encode(s, add_special_tokens=False).ids) == s for s in xs + en)
        rows.append(dict(lang=code, model=short, arm=arm, K=int(K), premium=float(c_x.sum() / c_en.sum()), lo=lo,
                         hi=hi, lang_tokens=int(c_x.sum()), en_tokens=int(c_en.sum()), floor_pretokens=floor,
                         floor_premium=floor / E, en_identical=same, lossless=lossless,
                         new_merges=meta["new_merges"], new_tokens=meta["new_tokens"],
                         train_bytes=meta["bytes"], train_types=meta["types"],
                         fragment_merges_total_run=meta.get("fragment_merges_total_run")))
        print(code, short, arm, K, round(rows[-1]["premium"], 4), same, lossless, flush=True)
    (H / "results/xl").mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(H / "results/xl" / f"{code}.{short}.json", "w"), indent=1)


def merge():
    rows = []
    for p in sorted((H / "results/xl").glob("*.json")):
        rows += json.load(open(p))
    prep = {p.name.split(".")[1]: json.load(open(p)) for p in (H / "data/xl").glob("prep.*.json")}
    out = dict(protocol=dict(
        eval="FLORES-200 devtest, NFC; premium = sum(lang tokens)/sum(eng tokens) over 1012 paired sentences; "
             "paired bootstrap 95% CI B=2000 seed 0; floor_premium = pre-tokens(lang, arm regex)/base English tokens",
        build="box/retrofit_tok_xl.py: continued BPE over base tokens, target-script merges only, K-prefix of one "
              "Kmax=32000 run per (lang, base, arm); FRAG rule on",
        train_text="FineWeb-2 TEST split only, NFC, exact-dedup, FLORES(lang+eng) whitespace-10-gram + char-50-gram "
                   "decontaminated; all surviving text used"),
        prep=prep, rows=rows)
    json.dump(out, open(H / "results/xling_ksweep.json", "w"), indent=1)
    print(len(rows), "rows")


if __name__ == "__main__":
    merge() if sys.argv[1] == "--merge" else run(sys.argv[1], sys.argv[2])
