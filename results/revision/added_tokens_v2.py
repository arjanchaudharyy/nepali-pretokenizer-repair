"""
added_tokens_v2.py (review R1 W3): the fair added-tokens arm.

R2_K32000's 32,000 new vocabulary entries (learned under the REPAIRED regex, so they are
whole words / multi-syllable strings) are decoded to text (byte-level -> UTF-8, strict).
Kept: strings that decode to valid UTF-8 and consist only of Devanagari-block characters
(U+0900-U+097F), optionally preceded by ONE leading space (the byte-level "G" prefix).
Those strings are registered with tokenizer.add_tokens(AddedToken(s, normalized=False))
on the ORIGINAL HF tokenizer (unchanged letters-only regex, unchanged merges) of
Llama-3.2-1B and Qwen3-1.7B-Base.
  AT2       single_word=False (HF default; leftmost-longest match anywhere in raw text)
  AT2_sw    single_word=True  (match only when not flanked by word characters)
Compared, on FLORES-200 devtest (1,012 sentences, NFC), with R0, R1_K32000, R2_K32000
and R1b (R1 strings as added tokens, as in tokenizer_extras.py).
Premium = Nepali tokens / English tokens of the same tokenizer, paired bootstrap 95% CI
(B=2000, seed 0, as box/ksweep_eval.py / tokenizer_extras.py). English identity = English
ids equal to the base tokenizer on all 1,012 sentences; losslessness = decode(encode(x))==x
on all Nepali and English sentences (tokenizers.Tokenizer.decode). Also: ids unchanged on
every sentence of every other FLORES language with no Devanagari character, mark-initial
rate (tokenizer_extras.mark_initial) and greedy-match pathologies.

Run from the repo root:  .venv/bin/python results/revision/added_tokens_v2.py
Output: results/revision/added_tokens_v2.json
"""
import json
import sys
import time
from pathlib import Path
import regex
from tokenizers import Tokenizer, AddedToken
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tokenizer_extras import load, compress, mark_initial, build_r1b, U2B  # noqa: E402

BASES = {"Llama-3.2-1B": "unsloth/Llama-3.2-1B", "Qwen3-1.7B-Base": "Qwen/Qwen3-1.7B-Base"}
DEVA_ONLY = regex.compile(r"^ ?[ऀ-ॿ]+$")
DEVA_ANY = regex.compile(r"[ऀ-ॿ]")
DEVA_CH = regex.compile(r"[ऀ-ॿ]")


def r2_strings(r2_dir):
    meta = json.load(open(r2_dir / "meta.json"))
    r2 = Tokenizer.from_file(str(r2_dir / "tokenizer.json"))
    first = meta["first_new_id"]
    keep, partial, other = [], 0, []
    for i in range(first, first + meta["new_tokens"]):
        raw = bytes(U2B[c] for c in r2.id_to_token(i))
        try:
            s = raw.decode("utf-8")
        except UnicodeDecodeError:
            partial += 1
            continue
        (keep if DEVA_ONLY.match(s) else other).append(s)
    return keep, dict(candidates=meta["new_tokens"], partial_utf8=partial, not_pure_devanagari=len(other),
                      not_pure_examples=other[:10], kept=len(keep),
                      kept_leading_space=sum(s.startswith(" ") for s in keep),
                      kept_mean_chars=sum(len(s.strip()) for s in keep) / len(keep))


def build(repo, strs, single_word):
    hf = AutoTokenizer.from_pretrained(repo)
    n0 = len(hf)
    added = hf.add_tokens([AddedToken(s, normalized=False, single_word=single_word) for s in strs])
    return hf.backend_tokenizer, dict(added=added, vocab_before=n0, vocab_after=len(hf), first_added_id=n0)


def pathologies(tk, sents, encs, first_added):
    n_tok = n_add = mid_start = mid_end = lone_space = 0
    for s, e in zip(sents, encs):
        for i, (a, b) in zip(e.ids, e.offsets):
            n_tok += 1
            if tk.id_to_token(i) in ("Ġ", " "):
                lone_space += 1
            if i >= first_added:
                n_add += 1
                if a > 0 and DEVA_CH.match(s[a - 1]) and s[a] != " ":
                    mid_start += 1
                if b < len(s) and DEVA_CH.match(s[b]):
                    mid_end += 1
    return dict(tokens=n_tok, added_token_share=n_add / n_tok,
                added_starting_inside_word=mid_start / max(n_add, 1),
                added_ending_inside_word=mid_end / max(n_add, 1), lone_space_tokens=lone_space)


def main():
    t0 = time.time()
    ne, en = load("npi_Deva"), load("eng_Latn")
    others = {p.name.split(".")[0]: load(p.name.split(".")[0]) for p in (ROOT / "flores200_dataset/devtest").glob("*.devtest")}
    res = dict(models={})
    for short, repo in BASES.items():
        R = dict(compression={}, mark_initial={}, pathologies={})
        base = AutoTokenizer.from_pretrained(repo).backend_tokenizer
        import numpy as np
        b_en = np.array([len(e.ids) for e in base.encode_batch(en, add_special_tokens=False)])
        d1, d2 = ROOT / "tok/box" / short / "R1_K32000", ROOT / "tok/box" / short / "R2_K32000"
        strs, R["strings"] = r2_strings(d2)
        toks, first = {"R0": base, "R1_K32000": Tokenizer.from_file(str(d1 / "tokenizer.json")),
                       "R2_K32000": Tokenizer.from_file(str(d2 / "tokenizer.json"))}, {}
        toks["R1b_added"], _ = build_r1b(repo, d1)
        toks["AT2"], R["build_AT2"] = build(repo, strs, False)
        toks["AT2_sw"], R["build_AT2_sw"] = build(repo, strs, True)
        first = {"AT2": R["build_AT2"]["first_added_id"], "AT2_sw": R["build_AT2_sw"]["first_added_id"]}
        for name, tk in toks.items():
            R["compression"][name], enc = compress(tk, base, ne, en, b_en)
            R["mark_initial"][name] = mark_initial(tk, ne, enc)
            if name in first:
                R["pathologies"][name] = pathologies(tk, ne, enc, first[name])
            print(short, name, json.dumps(R["compression"][name]), R["mark_initial"][name],
                  R["pathologies"].get(name, ""), flush=True)
        # invariance on every FLORES language for the AT2 arms
        for name in ("AT2", "AT2_sw"):
            ch_nodeva, ch_deva, n_nodeva = 0, {}, 0
            for c, sents in others.items():
                eb = base.encode_batch(sents, add_special_tokens=False)
                ea = toks[name].encode_batch(sents, add_special_tokens=False)
                for s, a, b in zip(sents, eb, ea):
                    if a.ids != b.ids:
                        if DEVA_ANY.search(s):
                            ch_deva[c] = ch_deva.get(c, 0) + 1
                        else:
                            ch_nodeva += 1
                    if not DEVA_ANY.search(s):
                        n_nodeva += 1
            R[f"invariance_{name}"] = dict(sentences_without_devanagari=n_nodeva,
                                           changed_without_devanagari=ch_nodeva,
                                           changed_with_devanagari_per_lang=ch_deva)
            print(short, name, R[f"invariance_{name}"], flush=True)
        res["models"][short] = R
    res["secs"] = time.time() - t0
    json.dump(res, open(ROOT / "results/revision/added_tokens_v2.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
