"""Build and verify the Hugging Face tokenizer releases (exactly the K=32,000 tokenizers used in the paper)."""
import json, shutil, unicodedata
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
P = ROOT.parent
FL = P / "flores200_dataset" / "devtest"
GH = "https://github.com/arjanchaudharyy/nepali-pretokenizer-repair"
L = lambda c: [unicodedata.normalize("NFC", x) for x in (FL / f"{c}.devtest").read_text().splitlines()]
NE, EN = L("npi_Deva"), L("eng_Latn")

REL = [  # (hf repo name, base repo, local tokenizer dir, arm, family)
    ("Llama-3.2-Nepali-Repaired-Tokenizer-32k", "unsloth/Llama-3.2-1B", "Llama-3.2-1B/R2_K32000", "R2", "llama"),
    ("Llama-3.2-Nepali-Extended-Tokenizer-32k", "unsloth/Llama-3.2-1B", "Llama-3.2-1B/R1_K32000", "R1", "llama"),
    ("Qwen3-Nepali-Repaired-Tokenizer-32k", "Qwen/Qwen3-1.7B-Base", "Qwen3-1.7B-Base/R2_K32000", "R2", "qwen"),
    ("Qwen3-Nepali-Extended-Tokenizer-32k", "Qwen/Qwen3-1.7B-Base", "Qwen3-1.7B-Base/R1_K32000", "R1", "qwen"),
]

out = {}
for name, base_repo, tokdir, arm, fam in REL:
    d = ROOT / name
    if d.exists():
        shutil.rmtree(d)
    base = AutoTokenizer.from_pretrained(base_repo)
    base.save_pretrained(d)  # configs + special tokens of the base model
    shutil.copy(P / "tok/box" / tokdir / "tokenizer.json", d / "tokenizer.json")
    shutil.copy(P / "tok/box" / tokdir / "meta.json", d / "build_meta.json")
    if fam == "llama":  # Llama 3.2 Community License s.1.b.iii: attribution in a "Notice" text file
        (d / "NOTICE").write_text("Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright \u00a9 Meta Platforms, Inc. All Rights Reserved.\n")
    for f in ("vocab.json", "merges.txt"):  # slow-tokenizer files from the base would be stale
        (d / f).unlink(missing_ok=True)
    # Model-specific classes (e.g. transformers' Qwen2Tokenizer) rebuild the pre-tokenizer from a
    # hard-coded pattern and would silently undo the repair. Use the generic fast class.
    cfg = json.load(open(d / "tokenizer_config.json"))
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    json.dump(cfg, open(d / "tokenizer_config.json", "w"), indent=2, ensure_ascii=False)
    tk = AutoTokenizer.from_pretrained(d)
    ids = lambda t, s: t(s, add_special_tokens=False)["input_ids"]
    en_same = all(ids(tk, s) == ids(base, s) for s in EN)
    lossless = all(tk.decode(ids(tk, s)) == s for s in NE + EN)
    specials_same = all(tk.convert_tokens_to_ids(t) == base.convert_tokens_to_ids(t) for t in base.all_special_tokens)
    ne_t, en_t = sum(len(ids(tk, s)) for s in NE), sum(len(ids(tk, s)) for s in EN)
    ne_b = sum(len(ids(base, s)) for s in NE)
    meta = json.load(open(d / "build_meta.json"))
    rx = json.loads(tk.backend_tokenizer.to_str())["pre_tokenizer"]
    out[name] = dict(regex_as_loaded_matches=(meta["regex"] in json.dumps(rx, ensure_ascii=False).replace("\\\\", "\\")) or (meta["regex"] in json.dumps(rx)) , base=base_repo, arm=arm, vocab=len(tk), new_tokens=meta["new_tokens"], english_identical=en_same,
                     lossless=lossless, special_ids_identical=specials_same, premium=ne_t / en_t,
                     base_premium=ne_b / en_t, regex=meta["regex"])
    print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items() if k != "regex"}))
json.dump(out, open(ROOT / "verification.json", "w"), indent=1, ensure_ascii=False)
assert all(v["english_identical"] and v["lossless"] and v["special_ids_identical"] for v in out.values()), "verification failed"
exp = {"R2": 1.02, "R1": 2.21}
assert all(v["premium"] < exp[v["arm"]] for v in out.values()), "premium does not match the paper: regex overridden?"
print("ALL VERIFIED")
