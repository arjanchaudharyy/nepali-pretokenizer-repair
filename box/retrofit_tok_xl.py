"""
retrofit_tok_xl.py: cross-lingual generalization of retrofit_tok.py (which is
left unchanged and remains the Nepali reference). Same continued-BPE procedure,
same R1/R2 arms, same English-safety construction; differences:

  1. The target-script filter is a parameter (`script`, a key of SCRIPTS: a
     Unicode-block character class), instead of hardcoded Devanagari.
  2. Byte-fragment merges are admitted (FRAG=1, default). The original filter
     keeps a merge only if its decoded output contains a complete target-script
     character. For scripts whose characters are NOT single tokens in the base
     vocab (e.g. Telugu/Sinhala/Burmese/Amharic in Llama-3, where >98% of
     FLORES character occurrences are multi-token), the first merge that joins
     two UTF-8 bytes of one character yields a fragment, is discarded, and the
     character can then never be formed: continued BPE would be crippled for a
     reason unrelated to the pre-tokenizer. With FRAG=1 a merge is also kept if
     every non-ASCII byte run in its output parses as target-script UTF-8
     fragments ([proper suffix, only at token start][complete target chars]
     [proper prefix, only at token end]); ASCII bytes such as the word's leading
     space may surround them (" " + lead-byte fragment is the typical case: in
     Llama-3 Sinhala, dropping it killed every space-initial word). Such tokens
     contain a non-ASCII byte so cannot match ASCII text; English identity is
     still verified on all FLORES English devtest sentences. Set FRAG=0 in the environment for the original rule.
  3. One training run per (base, arm, script) with the largest K; tokenizers
     for smaller K are the K-prefix of the filtered merge list. Greedy BPE emits
     merges in the same order regardless of the target vocab size, so the
     K-prefix is what a separate run with that K would produce (up to the
     trainer's own tie-breaking, which is not deterministic across runs anyway).

usage: python retrofit_tok_xl.py <base_repo> <arm R1|R2> <K1,K2,...> <script> <text> <bytes> <outroot>
writes <outroot>/<arm>_K<k>/{tokenizer.json,meta.json}
"""
import collections, json, os, sys, time, unicodedata
from pathlib import Path
import regex
from tokenizers import Tokenizer, trainers, models
from tokenizers.pre_tokenizers import WhitespaceSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrofit_tok import PAT_B, PUA, base_regex, repaired, set_regex  # noqa: E402  (single source of truth)

# Unicode blocks per script (main block + extension blocks where they exist).
SCRIPTS = {
    "Deva": r"[ऀ-ॿ꣠-ꣿ]",
    "Beng": r"[ঀ-৿]",
    "Taml": r"[஀-௿\U00011FC0-\U00011FFF]",
    "Telu": r"[ఀ-౿]",
    "Sinh": r"[඀-෿\U000111E0-\U000111FF]",
    "Thai": r"[฀-๿]",
    "Mymr": r"[က-႟ꧠ-꧿ꩠ-ꩿ]",
    "Ethi": r"[ሀ-᎟ⶀ-⷟꬀-꬯\U0001E7E0-\U0001E7FF]",
    "Hang": r"[ᄀ-ᇿ㄰-㆏ꥠ-꥿가-힯ힰ-퟿]",
}


def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def make_filter(script, frag):
    rx = regex.compile(SCRIPTS[script])
    lo_hi = []  # enumerate every code point in the class to build fragment sets
    for cp in range(0x80, 0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        if rx.match(chr(cp)):
            lo_hi.append(chr(cp).encode())
    pref = {e[:i] for e in lo_hi for i in range(1, len(e))}
    suf = {e[i:] for e in lo_hi for i in range(1, len(e))}
    bdec = byte_decoder()

    def ok(tokstr):
        b = bytes(bdec[c] for c in tokstr)
        s = b.decode("utf-8", errors="replace")
        if rx.search(s):
            return True  # original rule
        if not frag or all(x < 0x80 for x in b):
            return False
        # Every maximal non-ASCII run must be target-char fragments: a proper
        # suffix (only if the run starts the token), then complete target chars,
        # then a proper prefix (only if the run ends the token). ASCII bytes
        # (e.g. the leading space a word pre-token carries) are allowed around
        # them; a token with any non-ASCII byte can never match ASCII text.
        k, n = 0, len(b)
        while k < n:
            if b[k] < 0x80:
                k += 1
                continue
            j = k
            while j < n and b[j] >= 0x80:
                j += 1
            r = b[k:j]
            i = next((q for q, x in enumerate(r) if x >= 0xC0), len(r))
            if i and not (k == 0 and r[:i] in suf):
                return False
            q = i
            while q < len(r):
                L = 2 if r[q] < 0xE0 else 3 if r[q] < 0xF0 else 4
                ch = r[q:q + L]
                if len(ch) == L and all(0x80 <= x < 0xC0 for x in ch[1:]):
                    if not rx.match(ch.decode("utf-8", errors="replace")):
                        return False
                    q += L
                else:
                    if not (j == n and r[q:] in pref):
                        return False
                    q = len(r)
            k = j
        return True
    return rx, ok


def main(base_repo, arm, Ks, script, text_path, nbytes, outroot):
    t0 = time.time()
    Ks = sorted(int(k) for k in Ks.split(","))
    Kmax, nbytes = Ks[-1], int(float(nbytes))
    frag = os.environ.get("FRAG", "1") == "1"
    SCR, keep = make_filter(script, frag)
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(base_repo)
    base = hf.backend_tokenizer
    cfg = json.loads(base.to_str())
    pat_a = base_regex(cfg)
    pat = pat_a if arm == "R1" else repaired(pat_a)
    rx = regex.compile(pat)

    counts = collections.Counter()
    got = 0
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            got += len(line.encode())
            for w in rx.findall(unicodedata.normalize("NFC", line)):
                if SCR.search(w):
                    counts[w] += 1
            if got >= nbytes:
                break
    print(f"types={len(counts):,} tokens={sum(counts.values()):,} ({time.time()-t0:.0f}s)", flush=True)

    seg_cfg = json.loads(base.to_str())
    seg_cfg["pre_tokenizer"] = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False,
                                "use_regex": False}
    seg_cfg["normalizer"] = None
    seg = Tokenizer.from_str(json.dumps(seg_cfg))
    types = list(counts)
    enc = seg.encode_batch(types, add_special_tokens=False)
    used = sorted({i for e in enc for i in e.ids})
    assert len(used) <= len(PUA)
    id2pua = {i: chr(PUA[k]) for k, i in enumerate(used)}
    pua2id = {v: k for k, v in id2pua.items()}
    base_vocab = cfg["model"]["vocab"]
    inv = {v: k for k, v in base_vocab.items()}

    pairs = [("".join(id2pua[i] for i in e.ids), counts[w]) for e, w in zip(enc, types)]
    del enc, types

    def it():
        for s, c in pairs:
            while c > 0:
                m = min(c, 512)
                yield (" " + s) * m
                c -= m

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = WhitespaceSplit()
    slack = Kmax // 4 + 4000
    tr = trainers.BpeTrainer(vocab_size=len(used) + Kmax + slack, min_frequency=2, show_progress=False,
                             initial_alphabet=[id2pua[i] for i in used], limit_alphabet=len(used) + 10)
    tk.train_from_iterator(it(), trainer=tr)
    learned = json.loads(tk.to_str())["model"]["merges"]
    learned = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in learned]
    del tk, pairs

    def to_str(p):
        return "".join(inv[pua2id[ch]] for ch in p)

    alive, new_merges, new_tokens, n_frag = set(id2pua.values()), [], [], 0
    seen_tok, n_dead, n_filt, filt_ex = set(), 0, 0, collections.Counter()
    for a, b in learned:
        if a not in alive or b not in alive:
            n_dead += 1
            continue
        sa, sb = to_str(a), to_str(b)
        merged = sa + sb
        if not keep(merged):
            n_filt += 1
            filt_ex[repr(byte_decoded(merged))] += 1
            continue
        if not SCR.search(byte_decoded(merged)):
            n_frag += 1
        alive.add(a + b)
        new_merges.append([sa, sb])
        if merged not in base_vocab and merged not in seen_tok:
            seen_tok.add(merged)
            new_tokens.append((len(new_merges), merged))  # (merge index it first appears at, token)
        if len(new_merges) >= Kmax:
            break
    print(f"learned={len(learned):,} kept merges={len(new_merges):,} fragment merges={n_frag:,} dropped_filter={n_filt:,} dropped_dependent={n_dead:,} ({time.time()-t0:.0f}s)",
          flush=True)
    print("filter-dropped examples:", filt_ex.most_common(40), flush=True)

    for K in Ks:
        nm = new_merges[:K]
        nt = [t for k, t in new_tokens if k <= K]
        ext = json.loads(base.to_str())
        if arm == "R2":
            set_regex(ext, pat)
        vocab = ext["model"]["vocab"]
        for t in ext.get("added_tokens", []):
            vocab.setdefault(t["content"], t["id"])
        top = max(vocab.values())
        for k, t in enumerate(nt):
            vocab[t] = top + 1 + k
        m = ext["model"]["merges"]
        as_str = bool(m) and isinstance(m[0], str)
        ext["model"]["merges"] = m + ([" ".join(x) for x in nm] if as_str else nm)
        final = Tokenizer.from_str(json.dumps(ext))
        for t in ext.get("added_tokens", []):
            assert final.token_to_id(t["content"]) == t["id"], t
        for k, t in enumerate(nt):
            assert final.token_to_id(t) == top + 1 + k
        out = Path(outroot) / f"{arm}_K{K}"
        out.mkdir(parents=True, exist_ok=True)
        final.save(str(out / "tokenizer.json"))
        meta = dict(base=base_repo, arm=arm, K=K, script=script, frag_rule=frag, bytes=got, types=len(counts),
                    regex=pat, base_regex=pat_a, new_merges=len(nm), new_tokens=len(nt),
                    fragment_merges_total_run=n_frag, dropped_filter_run=n_filt, dropped_dependent_run=n_dead, learned_run=len(learned), first_new_id=top + 1, secs=round(time.time() - t0, 1))
        json.dump(meta, open(out / "meta.json", "w"), indent=1, ensure_ascii=False)
        print(json.dumps(meta, ensure_ascii=False), flush=True)


_BDEC = byte_decoder()


def byte_decoded(tokstr):
    return bytes(_BDEC[c] for c in tokstr).decode("utf-8", errors="replace")


if __name__ == "__main__":
    main(*sys.argv[1:8])
