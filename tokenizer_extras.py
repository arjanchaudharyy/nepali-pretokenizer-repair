"""
tokenizer_extras.py: CPU-only tokenizer diagnostics requested by review/REVIEW_1.md
(W4, W6, W10, N7). Writes results/tokenizer_extras.json.

For each base model (Llama-3.2-1B, Qwen3-1.7B-Base) on FLORES-200 devtest:

 1. R2_K0  regex-only arm: base tokenizer.json with its Split regex replaced by
           retrofit_tok.repaired(); vocab and merges untouched.
 2. R1b    added-tokens arm: the R1_K32000 new token strings (byte-level vocab
           entries with id >= first_new_id), decoded to text and registered via
           tokenizer.add_tokens(AddedToken(s, normalized=False)) on the BASE
           tokenizer (original regex, no new merges). Tokens whose byte string is
           not valid UTF-8 (partial code points) cannot be expressed as text and
           are dropped (count reported).
 3. mark-initial rate for R0, R1_K32000, R2_K32000, R2_K0 (and R1b): share of
    Nepali tokens whose first character is a combining mark (\\p{M}); computed
    from each token's exact byte span (incremental byte decoding), so a token
    that starts mid-code-point is never counted as mark-initial. Reported over
    all tokens and over Devanagari-bearing tokens.
 4. unseen rate: share of FLORES devtest Nepali tokens / token bigrams (within a
    sentence) under R2_K0 and R2_K32000 whose id / id pair never occurs in the
    BASE tokenizer's segmentation of the first 30 MB of data/ne_test_sample.txt.
    R0 and R1_K32000 on FLORES are given as controls (sampling noise floor).
 5. fragment inventory: distinct Devanagari-bearing pre-token types in
    data/ne_test_sample.txt (first 30 MB and full file) under the original vs
    repaired regex.

Inputs: tok/box/<model>/R{1,2}_K32000/{tokenizer.json,meta.json} copied from the
training box; results/ksweep_meta/ (all K-sweep meta.json files) is summarised
too (build secs).
"""
import collections, glob, json, sys, time, unicodedata
from pathlib import Path
import numpy as np
import regex
from tokenizers import Tokenizer, AddedToken
from transformers import AutoTokenizer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "box"))
from retrofit_tok import base_regex, repaired, set_regex  # noqa: E402

FL = ROOT / "flores200_dataset"
SAMPLE = ROOT / "data/ne_test_sample.txt"
BASES = {"Llama-3.2-1B": "unsloth/Llama-3.2-1B", "Qwen3-1.7B-Base": "Qwen/Qwen3-1.7B-Base"}
DEVA = regex.compile(r"[ऀ-ॿ]")
MARK = regex.compile(r"\p{M}")
UNSEEN_BYTES = 30_000_000


def load(code, split="devtest"):
    return [unicodedata.normalize("NFC", x) for x in (FL / split / f"{code}.{split}").read_text().splitlines()]


def boot(a, b, B=2000):  # identical to box/ksweep_eval.py
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    r = a[idx].sum(1) / b[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def read_sample(nbytes=None):
    out, got = [], 0
    with open(SAMPLE, encoding="utf-8") as f:
        for line in f:
            got += len(line.encode())
            out.append(unicodedata.normalize("NFC", line))
            if nbytes and got >= nbytes:
                break
    return out, got


# ---------------------------------------------------------------- byte spans
def byte_map():
    # GPT-2 bytes_to_unicode inverse
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


U2B = byte_map()


def token_bytes_table(tk):
    """id -> raw bytes. Model-vocab tokens are byte-level strings; added tokens
    (normalized=False, R1b) are literal text."""
    added = {i: t.content for i, t in tk.get_added_tokens_decoder().items()}
    tab = {}
    for s, i in tk.get_vocab(with_added_tokens=False).items():
        tab[i] = bytes(U2B[c] for c in s)
    for i, s in added.items():
        tab[i] = s.encode()
    return tab


def mark_initial(tk, sents, encs):
    tab = token_bytes_table(tk)
    n_all = n_dev = mi_all = mi_dev = 0
    for s, e in zip(sents, encs):
        b = s.encode()
        assert b"".join(tab[i] for i in e.ids) == b, "byte spans do not reconstruct the sentence"
        cob, starts, ci = [], set(), -1  # char index owning each byte; code-point start offsets
        for bi in range(len(b)):
            if (b[bi] & 0xC0) != 0x80:
                ci += 1
                starts.add(bi)
            cob.append(ci)
        pos = 0
        for i in e.ids:
            st, en = pos, pos + len(tab[i])
            pos = en
            if en == st:
                continue
            seg = s[cob[st]:cob[en - 1] + 1]  # chars touched (incl. partial code points)
            dev = bool(DEVA.search(seg))
            mi = st in starts and bool(MARK.match(s[cob[st]]))
            n_all += 1
            mi_all += mi
            if dev:
                n_dev += 1
                mi_dev += mi
    return dict(all=mi_all / n_all, devanagari=mi_dev / n_dev, n_tokens=n_all, n_deva_tokens=n_dev)


# ---------------------------------------------------------------- builders
def build_r2k0(base):
    cfg = json.loads(base.to_str())
    pat_a = base_regex(cfg)
    set_regex(cfg, repaired(pat_a))
    return Tokenizer.from_str(json.dumps(cfg))


def build_r1b(repo, r1_dir):
    meta = json.load(open(r1_dir / "meta.json"))
    r1 = Tokenizer.from_file(str(r1_dir / "tokenizer.json"))
    first = meta["first_new_id"]
    strs, bad = [], 0
    for i in range(first, first + meta["new_tokens"]):
        raw = bytes(U2B[c] for c in r1.id_to_token(i))
        try:
            strs.append(raw.decode("utf-8"))
        except UnicodeDecodeError:
            bad += 1
    hf = AutoTokenizer.from_pretrained(repo)
    n_before = len(hf)
    added = hf.add_tokens([AddedToken(s, normalized=False) for s in strs])
    tk = hf.backend_tokenizer
    return tk, dict(candidates=meta["new_tokens"], partial_utf8_dropped=bad, added=added,
                    already_in_vocab_or_dup=len(strs) - added, vocab_before=n_before, vocab_after=len(hf),
                    leading_space=sum(s.startswith(" ") for s in strs))


# ---------------------------------------------------------------- main
def compress(tk, base, ne, en, b_en):
    e_ne = tk.encode_batch(ne, add_special_tokens=False)
    c_ne = np.array([len(e.ids) for e in e_ne])
    c_en = np.array([len(e.ids) for e in tk.encode_batch(en, add_special_tokens=False)])
    lo, hi = boot(c_ne, c_en)
    same = all(base.encode(s, add_special_tokens=False).ids == tk.encode(s, add_special_tokens=False).ids for s in en)
    lossless_ne = all(tk.decode(e.ids) == s for s, e in zip(ne, e_ne))
    lossless_en = all(tk.decode(tk.encode(s, add_special_tokens=False).ids) == s for s in en)
    return dict(premium=float(c_ne.sum() / c_en.sum()), lo=lo, hi=hi, ne_tokens=int(c_ne.sum()),
                en_tokens=int(c_en.sum()), premium_vs_base_en=float(c_ne.sum() / b_en.sum()),
                en_identical=same, lossless_ne=lossless_ne, lossless_en=lossless_en), e_ne


def seq_stats(encs):
    toks, bigs = collections.Counter(), collections.Counter()
    for e in encs:
        ids = e.ids
        toks.update(ids)
        bigs.update(zip(ids, ids[1:]))
    return toks, bigs


def unseen(encs, seen_t, seen_b, first_new):
    toks, bigs = seq_stats(encs)
    nt, nb = sum(toks.values()), sum(bigs.values())
    old_b = {k: v for k, v in bigs.items() if k[0] < first_new and k[1] < first_new}
    nob = sum(old_b.values())
    return dict(
        tokens_unseen=sum(v for k, v in toks.items() if k not in seen_t) / nt,
        bigrams_unseen=sum(v for k, v in bigs.items() if k not in seen_b) / nb,
        old_tokens_unseen=sum(v for k, v in toks.items() if k < first_new and k not in seen_t)
        / max(1, sum(v for k, v in toks.items() if k < first_new)),
        old_old_bigrams_unseen=sum(v for k, v in old_b.items() if k not in seen_b) / max(1, nob),
        old_old_bigram_share=nob / nb,
        token_types_unseen=sum(k not in seen_t for k in toks) / len(toks),
        bigram_types_unseen=sum(k not in seen_b for k in bigs) / len(bigs),
        n_tokens=nt, n_bigrams=nb)


def fragments(lines_full, nbytes30, pats):
    out = {}
    for name, pat in pats.items():
        rx = regex.compile(pat)
        types, toks, got, at30 = collections.Counter(), 0, 0, None
        for line in lines_full:
            got += len(line.encode())
            for w in rx.findall(line):
                if DEVA.search(w):
                    types[w] += 1
            if at30 is None and got >= nbytes30:
                at30 = dict(bytes=got, types=len(types), tokens=sum(types.values()))
        tot = sum(types.values())
        freq = sorted(types.values(), reverse=True)
        cum = np.cumsum(freq) / tot
        mark_init_types = sum(1 for w in types if MARK.match(w.lstrip()))
        out[name] = dict(first_30MB=at30, full=dict(bytes=got, types=len(types), tokens=tot,
                         types_for_99pct=int(np.searchsorted(cum, 0.99) + 1),
                         types_for_999pct=int(np.searchsorted(cum, 0.999) + 1),
                         mark_initial_types=mark_init_types))
    return out


def main():
    t0 = time.time()
    ne, en = load("npi_Deva"), load("eng_Latn")
    res = dict(split="devtest", models={})
    print("reading sample", flush=True)
    sample30, got30 = read_sample(UNSEEN_BYTES)
    for short, repo in BASES.items():
        R = {}
        base = AutoTokenizer.from_pretrained(repo).backend_tokenizer
        b_en = np.array([len(e.ids) for e in base.encode_batch(en, add_special_tokens=False)])
        d1, d2 = ROOT / "tok/box" / short / "R1_K32000", ROOT / "tok/box" / short / "R2_K32000"
        first_new = json.load(open(d2 / "meta.json"))["first_new_id"]
        toks = {"R0": base, "R2_K0": build_r2k0(base),
                "R1_K32000": Tokenizer.from_file(str(d1 / "tokenizer.json")),
                "R2_K32000": Tokenizer.from_file(str(d2 / "tokenizer.json"))}
        r1b, r1b_info = build_r1b(repo, d1)
        toks["R1b_added32000"] = r1b
        R["r1b_build"] = r1b_info
        encs = {}
        R["compression"] = {}
        for name, tk in toks.items():
            R["compression"][name], encs[name] = compress(tk, base, ne, en, b_en)
            print(short, name, json.dumps(R["compression"][name]), flush=True)
        R["mark_initial"] = {}
        for name, tk in toks.items():
            R["mark_initial"][name] = mark_initial(tk, ne, encs[name])
            print(short, name, "mark-initial", R["mark_initial"][name], flush=True)
        # unseen rates vs base segmentation of 30 MB sample
        seen_t, seen_b = seq_stats(base.encode_batch([s.rstrip("\n") for s in sample30], add_special_tokens=False))
        R["unseen_ref"] = dict(sample_bytes=got30, sample_lines=len(sample30), base_token_types=len(seen_t),
                               base_bigram_types=len(seen_b))
        R["unseen"] = {n: unseen(encs[n], seen_t, seen_b, first_new) for n in ("R0", "R2_K0", "R1_K32000", "R2_K32000")}
        for n, v in R["unseen"].items():
            print(short, n, "unseen", v, flush=True)
        R["first_new_id"] = first_new
        res["models"][short] = R
    # fragment inventory (model-independent except via the regex; Llama/Qwen regex equal?)
    pats = {}
    for short, repo in BASES.items():
        pa = base_regex(json.loads(AutoTokenizer.from_pretrained(repo).backend_tokenizer.to_str()))
        pats[short] = pa
    same_regex = len(set(pats.values())) == 1
    print("reading full sample", flush=True)
    full, _ = read_sample(None)
    res["fragments"] = {}
    for short, pa in (list(pats.items())[:1] if same_regex else pats.items()):
        res["fragments"][short if not same_regex else "both"] = fragments(full, UNSEEN_BYTES,
                                                                          {"original": pa, "repaired": repaired(pa)})
    res["fragments_regex_identical_across_models"] = same_regex
    print("fragments", json.dumps(res["fragments"]), flush=True)
    # build-time summary from synced metas
    metas = [json.load(open(f)) for f in sorted(glob.glob(str(ROOT / "results/ksweep_meta/*/*/meta.json")))]
    k64 = [m["secs"] for m in metas if m["K"] == 64000]
    res["ksweep_meta"] = dict(n=len(metas), secs_K64000_min=min(k64), secs_K64000_max=max(k64),
                              secs_all_min=min(m["secs"] for m in metas), secs_all_max=max(m["secs"] for m in metas),
                              merge_text_bytes=sorted({m["bytes"] for m in metas}),
                              rows=[{k: m[k] for k in ("base", "arm", "K", "types", "new_tokens", "secs")} for m in metas])
    res["secs"] = round(time.time() - t0, 1)
    json.dump(res, open(ROOT / "results/tokenizer_extras.json", "w"), indent=1, ensure_ascii=False)
    print("done", res["secs"])


if __name__ == "__main__":
    main()
