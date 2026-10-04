"""
measure.py: token premium + akshara-integrity diagnostics on FLORES-200.

PREMIUM (Petrov et al. 2023): tokens(L) / tokens(eng) summed over the SAME
parallel sentences. Content is held constant by construction, so the ratio is
the price of the language, not of what is said. CIs: paired bootstrap over
sentences (B=2000, seed 0); a resample draws sentence indices once and uses
them for both languages.

AKSHARA INTEGRITY (Nepali-centric, but computed for every Indic/abugida set):
  mark_initial   share of tokens whose first character is a combining mark
                 (\\p{M}). A token cannot legitimately begin with a vowel sign;
                 every such token is a cluster cut in two.
  break_rate     share of token boundaries that fall INSIDE an orthographic
                 syllable: before a combining mark, or between a virama and the
                 consonant it joins.
  intra_cp       share of boundaries that fall inside a single code point
                 (byte-level BPE emitting partial UTF-8 sequences).
"""
from __future__ import annotations
import json, sys, time, unicodedata
from pathlib import Path
import numpy as np
import regex

ROOT = Path(__file__).parent
FL = ROOT / "flores200_dataset"

LANGS = [
    # Nepal
    "npi_Deva", "mai_Deva", "bho_Deva",
    # other Devanagari / Brahmic abugidas (vowel signs are combining marks)
    "hin_Deva", "mar_Deva", "ben_Beng", "guj_Gujr", "pan_Guru", "ory_Orya",
    "tam_Taml", "tel_Telu", "kan_Knda", "mal_Mlym", "sin_Sinh",
    "tha_Thai", "lao_Laoo", "mya_Mymr", "khm_Khmr", "bod_Tibt",
    # abugida WITHOUT combining marks (vowels precomposed into the letter): control
    "amh_Ethi",
    # non-Latin, non-abugida
    "arb_Arab", "heb_Hebr", "rus_Cyrl", "ell_Grek", "kor_Hang", "zho_Hans", "jpn_Jpan",
    # Latin
    "eng_Latn", "spa_Latn", "deu_Latn", "fra_Latn", "tur_Latn", "fin_Latn",
    "ind_Latn", "vie_Latn", "swh_Latn",
]

MARK = regex.compile(r"\p{M}")
VIRAMAS = {"्", "্", "੍", "્", "୍", "்", "్",
           "್", "്", "්", "္", "်", "្", "྄"}


def load_split(split: str, code: str) -> list[str]:
    p = FL / split / f"{code}.{split}"
    return [unicodedata.normalize("NFC", s) for s in p.read_text(encoding="utf-8").splitlines()]


def boundaries(tok, text: str, ids: list[int]):
    """Char offsets of token starts (excluding 0) + count of intra-codepoint cuts."""
    if tok.get("decode_bytes"):  # tiktoken: exact byte accounting
        bpos, starts, intra = 0, [], 0
        b = text.encode("utf-8")
        # char index for each byte offset that starts a code point
        cstart = {}
        ci = 0
        for bi in range(len(b)):
            if (b[bi] & 0xC0) != 0x80:
                cstart[bi] = ci
                ci += 1
        for k, i in enumerate(ids):
            if k > 0:
                if bpos in cstart:
                    starts.append(cstart[bpos])
                else:
                    intra += 1
            bpos += len(tok["decode_bytes"](i))
        return starts, intra
    off = tok["_offsets"](text)
    starts, intra, prev_s, prev_e = [], 0, None, None
    for (s, e) in off:
        if prev_s is not None:
            if s < prev_e or s == prev_s:
                intra += 1
            elif s > 0:
                starts.append(s)
        prev_s, prev_e = s, e
    return starts, intra


def integrity(tok, sents, ids_list):
    n_tok = n_bound = mark_init = brk = intra = 0
    for text, ids in zip(sents, ids_list):
        try:
            starts, ic = boundaries(tok, text, ids)
        except Exception:
            return None
        n_tok += len(ids)
        n_bound += len(starts) + ic
        intra += ic
        for s in starts:
            if s >= len(text):
                continue
            c, pc = text[s], text[s - 1]
            if MARK.match(c):
                mark_init += 1
                brk += 1
            elif pc in VIRAMAS and regex.match(r"\p{L}", c):
                brk += 1
    return dict(mark_initial=mark_init / max(n_tok, 1),
                break_rate=brk / max(n_bound, 1),
                intra_cp=intra / max(n_bound, 1))


def boot_ratio(a: np.ndarray, b: np.ndarray, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(B, n))
    r = a[idx].sum(1) / b[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def run(toks, split="devtest", langs=LANGS, integrity_langs=None):
    data = {c: load_split(split, c) for c in langs}
    integrity_langs = integrity_langs or [c for c in langs if c.split("_")[1] in
                                          ("Deva", "Beng", "Gujr", "Guru", "Orya", "Taml", "Telu",
                                           "Knda", "Mlym", "Sinh", "Thai", "Laoo", "Mymr", "Khmr", "Tibt")]
    rows = []
    for tok in toks:
        t0 = time.time()
        counts, lossless, integ = {}, {}, {}
        for c in langs:
            ids_list = [tok["encode"](s) for s in data[c]]
            counts[c] = np.array([len(x) for x in ids_list], dtype=np.int64)
            try:
                lossless[c] = all(tok["decode"](ids) == s for s, ids in zip(data[c][:200], ids_list[:200]))
            except Exception:
                lossless[c] = False
            if c in integrity_langs:
                integ[c] = integrity(tok, data[c], ids_list)
        en = counts["eng_Latn"]
        prem = {}
        for c in langs:
            lo, hi = boot_ratio(counts[c], en)
            prem[c] = dict(premium=float(counts[c].sum() / en.sum()), lo=lo, hi=hi,
                           tokens=int(counts[c].sum()),
                           bytes=sum(len(s.encode()) for s in data[c]),
                           chars=sum(len(s) for s in data[c]),
                           words=sum(len(s.split()) for s in data[c]))
        rows.append(dict(name=tok["name"], family=tok["family"], vocab=tok["vocab"],
                         kind=tok["kind"], regexes=tok["regexes"], premium=prem,
                         lossless=lossless, integrity=integ, secs=round(time.time() - t0, 1)))
        p = prem["npi_Deva"]
        print(f"  {tok['name']:<14} {tok['kind']:<13} V={tok['vocab']:>7}  ne/en={p['premium']:.3f} "
              f"[{p['lo']:.3f},{p['hi']:.3f}]  mark_init={integ.get('npi_Deva', {}) and integ['npi_Deva']['mark_initial']:.3f}"
              f"  ({rows[-1]['secs']}s)", flush=True)
    return rows


def attach_offsets(tok, repo=None):
    if tok.get("decode_bytes"):
        return tok
    if repo:
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(repo)
        tok["_offsets"] = lambda t, hf=hf: hf(t, add_special_tokens=False,
                                              return_offsets_mapping=True)["offset_mapping"]
    elif "_tk" in tok:
        tok["_offsets"] = lambda t, tk=tok["_tk"]: tk.encode(t, add_special_tokens=False).offsets
    return tok


if __name__ == "__main__":
    from tok_registry import load
    out = sys.argv[1] if len(sys.argv) > 1 else "results/sweep_devtest.json"
    toks, skipped = load()
    for t in toks:
        attach_offsets(t, t.get("repo"))
    print("loaded", len(toks), "skipped", skipped, flush=True)
    rows = run(toks)
    Path(out).parent.mkdir(exist_ok=True)
    json.dump(dict(split="devtest", rows=rows, skipped=skipped), open(out, "w"), ensure_ascii=False, indent=1)
