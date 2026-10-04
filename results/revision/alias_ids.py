"""
alias_ids.py (review R1 W7): verify the audit's alias groups (results/audit_unique.json)
at the token-ID level, not by equal token counts.

Pairs (canonical, alias): cl100k vs Phi-4 / OLMo-2 / Granite-4.1, Qwen2.5 vs Qwen3,
Llama-3 vs SmolLM3, o200k vs gpt-oss, Mistral-NeMo vs Sarvam-M, Gemma-3 vs Gemma-4,
DeepSeek-V3 vs DeepSeek-V4. Loaded with tok_registry.load (tokenizer files only,
trust_remote_code=False; tiktoken for cl100k/o200k; encode without special tokens).
FLORES-200 devtest, 1,012 sentences each (NFC, as measure.load_split), for eng_Latn,
npi_Deva, hin_Deva, zho_Hans, arb_Arab. Reported: sentences whose id sequence is
identical, total-token counts of each side, and first differing example.

Run from the repo root:  .venv/bin/python results/revision/alias_ids.py
Output: results/revision/alias_ids.json
"""
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tok_registry  # noqa: E402

PAIRS = [("cl100k", "Phi-4"), ("cl100k", "OLMo-2"), ("cl100k", "Granite-4.1"), ("Qwen2.5", "Qwen3"),
         ("Llama-3", "SmolLM3"), ("o200k", "gpt-oss"), ("Mistral-NeMo", "Sarvam-M"),
         ("Gemma-3", "Gemma-4"), ("DeepSeek-V3", "DeepSeek-V4")]
LANGS = ["eng_Latn", "npi_Deva", "hin_Deva", "zho_Hans", "arb_Arab"]
FL = ROOT / "flores200_dataset/devtest"

data = {c: [unicodedata.normalize("NFC", s) for s in (FL / f"{c}.devtest").read_text(encoding="utf-8").splitlines()]
        for c in LANGS}
names = sorted({n for p in PAIRS for n in p})
toks, skipped = tok_registry.load(names)
T = {t["name"]: t for t in toks}
out = dict(skipped=skipped, pairs={})
for a, b in PAIRS:
    if a not in T or b not in T:
        out["pairs"][f"{a}|{b}"] = dict(error="not loaded")
        continue
    r = dict(vocab=[T[a]["vocab"], T[b]["vocab"]], kinds=[T[a]["kind"], T[b]["kind"]],
             regexes_equal=T[a]["regexes"] == T[b]["regexes"], langs={})
    for c in LANGS:
        same, ta, tb, ex = 0, 0, 0, None
        for s in data[c]:
            ia, ib = list(T[a]["encode"](s)), list(T[b]["encode"](s))
            ta += len(ia)
            tb += len(ib)
            if ia == ib:
                same += 1
            elif ex is None:
                k = next((j for j, (x, y) in enumerate(zip(ia, ib)) if x != y), min(len(ia), len(ib)))
                ex = dict(sentence=s[:80], pos=k, a=ia[max(0, k - 2):k + 3], b=ib[max(0, k - 2):k + 3])
        r["langs"][c] = dict(identical=same, n=len(data[c]), tokens=[ta, tb], first_diff=ex)
    r["all_identical"] = all(v["identical"] == v["n"] for v in r["langs"].values())
    out["pairs"][f"{a}|{b}"] = r
    print(f"{a:>12} vs {b:<12} V={r['vocab']} all_identical={r['all_identical']} "
          + " ".join(f"{c}:{v['identical']}/{v['n']}" for c, v in r["langs"].items()), flush=True)
json.dump(out, open(ROOT / "results/revision/alias_ids.json", "w"), indent=1, ensure_ascii=False)
print("skipped:", skipped)
