"""
thai_burmese.py (review R1 W8): multi-word tokens in the cross-script sweep for Thai and
Burmese, which are written without spaces between words.

Tokenizers: tok/xl/{tha_Thai,mya_Mymr}/<model>/R{1,2}_K32000/tokenizer.json (the
cross-script sweep artifacts) and the HF base tokenizer (R0), for Llama-3.2-1B and
Qwen3-1.7B-Base. Text: FLORES-200 devtest (1,012 sentences, NFC).
Every token's exact text span is taken from the encoding offsets (tokens that start or end
inside a code point are attributed to the characters they touch).
  span_space        token text contains a space after its first character
                    (i.e. it crosses a space; a single leading space does not count)
  Thai: word boundaries from PyThaiNLP 5.3.8 word_tokenize(engine="newmm") on the sentence
        (installed only in the scratchpad, not in the project venv). multiword = the token
        strictly contains >= 1 internal newmm word boundary, i.e. spans >= 2 words.
  Burmese: no word segmenter is available offline, so syllables are used: syllable breaks
        from the standard sylbreak rule (Ye Kyaw Thu): a break before every consonant not
        preceded by U+1039 (stacking virama) and not followed by U+103A/U+1039, and before
        every digit/independent vowel/punctuation. Reported: syllables per token, share of
        tokens with >= 2, >= 3, >= 4 syllables; FLORES mean syllables per space-delimited
        chunk is given for scale. "Exceeds a typical word" cannot be measured directly.
Shares are over all tokens and over tokens that touch the target script.

Run from the repo root (PyThaiNLP on PYTHONPATH):
  PYTHONPATH=<scratchpad>/pylib .venv/bin/python results/revision/thai_burmese.py
Output: results/revision/thai_burmese.json
"""
import json
import unicodedata
from pathlib import Path
import numpy as np
import regex
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from pythainlp.tokenize import word_tokenize
import pythainlp

ROOT = Path(__file__).resolve().parents[2]
FL = ROOT / "flores200_dataset/devtest"
BASES = {"Llama-3.2-1B": "unsloth/Llama-3.2-1B", "Qwen3-1.7B-Base": "Qwen/Qwen3-1.7B-Base"}
SCRIPT = {"tha_Thai": regex.compile(r"[฀-๿]"), "mya_Mymr": regex.compile(r"[က-႟]")}
MY_CONS = "က-အ"
MY_BREAK = regex.compile(r"(?<!္)[" + MY_CONS + r"](?![်္])|[ဣ-ဧဩဪဿ၌-၏၀-။]")


def load(code):
    return [unicodedata.normalize("NFC", x) for x in (FL / f"{code}.devtest").read_text(encoding="utf-8").splitlines()]


def char_spans(tk, s):
    e = tk.encode(s, add_special_tokens=False)
    return [(a, b) for a, b in e.offsets if b > a], len(e.ids)


def thai_bounds(s):
    pos, out = 0, set()
    for w in word_tokenize(s, engine="newmm", keep_whitespace=True):
        pos += len(w)
        out.add(pos)
    return out


def my_bounds(s):
    return {m.start() for m in MY_BREAK.finditer(s)} | {len(s)}


def stats(tk, sents, code):
    sc = SCRIPT[code]
    n = n_sc = span_sp = span_sp_sc = 0
    units = []  # units (words or syllables) touched per script token
    multi = 0
    for s in sents:
        bnd = thai_bounds(s) if code == "tha_Thai" else my_bounds(s)
        spans, _ = char_spans(tk, s)
        for a, b in spans:
            t = s[a:b]
            n += 1
            sp = " " in t[1:]
            span_sp += sp
            if sc.search(t):
                n_sc += 1
                span_sp_sc += sp
                off = a + (len(t) - len(t.lstrip(" ")))  # ignore a leading space
                k = sum(1 for x in bnd if off < x < b)  # internal word / syllable boundaries
                multi += k >= 1
                units.append(k + 1)
    u = np.array(units)
    r = dict(tokens=n, script_tokens=n_sc, span_space_share=span_sp / n, span_space_share_script=span_sp_sc / max(n_sc, 1),
             mean_units_per_script_token=float(u.mean()))
    if code == "tha_Thai":
        r["multiword_share_script"] = multi / max(n_sc, 1)  # == share_ge2_words
        r["share_ge3_words"] = float((u >= 3).mean())
    else:
        for k in (2, 3, 4):
            r[f"share_ge{k}_syllables"] = float((u >= k).mean())
    return r


def main():
    out = dict(pythainlp=pythainlp.__version__, models={}, reference={})
    for code in SCRIPT:
        sents = load(code)
        if code == "tha_Thai":
            words = [w for s in sents for w in word_tokenize(s, engine="newmm", keep_whitespace=False) if SCRIPT[code].search(w)]
            out["reference"][code] = dict(newmm_words=len(words), mean_chars_per_word=float(np.mean([len(w) for w in words])),
                                          space_chunks=sum(len(s.split()) for s in sents))
        else:
            chunks = [c for s in sents for c in s.split() if SCRIPT[code].search(c)]
            sy = [len(my_bounds(c) - {0, len(c)}) + 1 for c in chunks]
            out["reference"][code] = dict(space_chunks=len(chunks), mean_syllables_per_space_chunk=float(np.mean(sy)),
                                          total_syllables=int(sum(sy)))
    for short, repo in BASES.items():
        for code in SCRIPT:
            d = ROOT / "tok/xl" / code / short
            toks = {"R0": AutoTokenizer.from_pretrained(repo).backend_tokenizer}
            for a in ("R1", "R2"):
                p = d / f"{a}_K32000" / "tokenizer.json"
                if p.exists():
                    toks[a] = Tokenizer.from_file(str(p))
            sents = load(code)
            for a, tk in toks.items():
                r = stats(tk, sents, code)
                r["total_tokens"] = sum(len(e.ids) for e in tk.encode_batch(sents, add_special_tokens=False))
                out["models"][f"{short}|{code}|{a}"] = r
                print(short, code, a, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    print(out["reference"])
    json.dump(out, open(ROOT / "results/revision/thai_burmese.json", "w"), indent=1)


if __name__ == "__main__":
    main()
