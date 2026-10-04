"""
retrofit_tok.py: extend a DEPLOYED byte-level BPE tokenizer for Nepali by
continued BPE, under either the model's own pre-tokenizer (R1) or the
mark-aware repair (R2). Nothing else differs between R1 and R2.

Continued BPE, exactly:
  1. Pre-tokenize Nepali text with the arm's regex; count pre-token types.
  2. Segment every type with the BASE tokenizer's own merges (so the starting
     point is exactly what the deployed model sees today).
  3. Learn K new merges greedily over those segmentations.
  4. Append the new merges AFTER the base merges. HF BPE applies the lowest-
     rank applicable merge first, so every word first reaches its base
     segmentation and only then sees new merges, in learned order. The
     extended tokenizer therefore reproduces step 3 exactly.

Step 3 runs inside the Rust BpeTrainer: each base token id used by the Nepali
segmentations is mapped to one private-use code point, so a word becomes a
string over an alphabet of base tokens, and an ordinary BPE run on those strings
IS continued BPE over base tokens.

ENGLISH SAFETY: only word types containing Devanagari are counted, and any merge
whose output has no Devanagari is discarded together with every later merge
that depends on it. English tokenization is therefore unchanged by
construction; `check_english` verifies it on real text anyway.

usage: python retrofit_tok.py <base_repo> <arm R1|R2> <K> <nepali_text> <bytes> <outdir>
"""
import collections, json, sys, time, unicodedata
from pathlib import Path
import regex
from tokenizers import Tokenizer, Regex, pre_tokenizers, trainers, models
from tokenizers.pre_tokenizers import WhitespaceSplit

PAT_B = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
DEVA = regex.compile(r"[ऀ-ॿ]")
PUA = [*range(0xE000, 0xF900), *range(0xF0000, 0xFFFFE), *range(0x100000, 0x10FFFE)]


def base_regex(cfg):
    pats = []

    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "Split" and "Regex" in n.get("pattern", {}):
                pats.append(n["pattern"]["Regex"])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(cfg["pre_tokenizer"])
    assert len(pats) == 1, pats
    return pats[0]


def repaired(pat_a: str) -> str:
    """Widen the letter word class to admit marks; exclude marks from the
    leading-char and punctuation classes. Asserted to match PAT_B for the
    Llama-3/Qwen family pattern so the repair is the minimal, documented edit."""
    b = pat_a.replace(r"[^\r\n\p{L}\p{N}]?\p{L}+", r"[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+") \
             .replace(r" ?[^\s\p{L}\p{N}]+", r" ?[^\s\p{L}\p{M}\p{N}]+")
    assert b != pat_a, "pattern did not change"
    return b


def set_regex(cfg, new):
    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "Split" and "Regex" in n.get("pattern", {}):
                n["pattern"]["Regex"] = new
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(cfg["pre_tokenizer"])


def main(base_repo, arm, K, text_path, nbytes, outdir):
    t0 = time.time()
    K, nbytes = int(K), int(float(nbytes))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(base_repo)
    base = hf.backend_tokenizer
    cfg = json.loads(base.to_str())
    pat_a = base_regex(cfg)
    pat = pat_a if arm == "R1" else repaired(pat_a)
    rx = regex.compile(pat)

    # 1. pre-token type counts on Nepali text
    counts = collections.Counter()
    got = 0
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            got += len(line.encode())
            for w in rx.findall(unicodedata.normalize("NFC", line)):
                if DEVA.search(w):
                    counts[w] += 1
            if got >= nbytes:
                break
    print(f"types={len(counts):,} tokens={sum(counts.values()):,} ({time.time()-t0:.0f}s)", flush=True)

    # 2. base segmentation of each type, via a tokenizer whose pre-tokenizer is
    #    disabled (the type IS one pre-token); byte-level mapping retained.
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

    # 3. continued BPE over base-token alphabet (Rust trainer)
    wfile = out / "words.txt"
    with open(wfile, "w", encoding="utf-8") as f:
        for e, w in zip(enc, types):
            s = "".join(id2pua[i] for i in e.ids)
            c = counts[w]
            # word-type file with repetition is wasteful; write "s\tc" and expand in iterator
            f.write(f"{s}\t{c}\n")

    def it():
        with open(wfile, encoding="utf-8") as f:
            for line in f:
                s, c = line.rstrip("\n").split("\t")
                # feed in chunks so the trainer's word counter sees the true frequency
                c = int(c)
                while c > 0:
                    m = min(c, 512)
                    yield (" " + s) * m
                    c -= m

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = WhitespaceSplit()
    tr = trainers.BpeTrainer(vocab_size=len(used) + K + 2000, min_frequency=2, show_progress=False,
                             initial_alphabet=[id2pua[i] for i in used], limit_alphabet=len(used) + 10)
    tk.train_from_iterator(it(), trainer=tr)
    learned = json.loads(tk.to_str())["model"]["merges"]
    learned = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in learned]

    # map back to base-token strings; drop non-Devanagari outputs and dependents
    def to_str(p):  # PUA string -> byte-level token string
        return "".join(inv[pua2id[ch]] for ch in p)

    from tokenizers.decoders import ByteLevel as BLD
    bld = BLD()
    alive, new_merges, new_tokens = set(id2pua.values()), [], []
    for a, b in learned:
        if a not in alive or b not in alive:
            continue
        sa, sb = to_str(a), to_str(b)
        merged = sa + sb
        if not DEVA.search(bld.decode([merged])):
            continue
        alive.add(a + b)
        new_merges.append([sa, sb])
        if merged not in base_vocab and merged not in new_tokens:
            new_tokens.append(merged)
        if len(new_merges) >= K:
            break
    print(f"new merges={len(new_merges):,} new tokens={len(new_tokens):,} ({time.time()-t0:.0f}s)", flush=True)

    # 4. assemble extended tokenizer: base vocab + new tokens appended after the
    #    current max id (added/special tokens keep their ids).
    ext = json.loads(base.to_str())
    if arm == "R2":
        set_regex(ext, pat)
    vocab = ext["model"]["vocab"]
    # Pin added/special tokens into the model vocab at their ORIGINAL ids. If
    # they are absent from model.vocab, tokenizers re-numbers them to
    # len(model.vocab) on load, which collides with the appended ids.
    for t in ext.get("added_tokens", []):
        vocab.setdefault(t["content"], t["id"])
    top = max(vocab.values())
    for k, t in enumerate(new_tokens):
        vocab[t] = top + 1 + k
    m = ext["model"]["merges"]
    as_str = bool(m) and isinstance(m[0], str)
    ext["model"]["merges"] = m + ([" ".join(x) for x in new_merges] if as_str else new_merges)
    final = Tokenizer.from_str(json.dumps(ext))
    for t in ext.get("added_tokens", []):  # ids must survive the round trip
        assert final.token_to_id(t["content"]) == t["id"], t
    for k, t in enumerate(new_tokens):
        assert final.token_to_id(t) == top + 1 + k and final.id_to_token(top + 1 + k) == t
    final.save(str(out / "tokenizer.json"))
    hf.save_pretrained(str(out / "hf"))  # copy configs, then overwrite tokenizer.json
    Tokenizer.from_str(json.dumps(ext)).save(str(out / "hf" / "tokenizer.json"))
    meta = dict(base=base_repo, arm=arm, K=K, bytes=got, types=len(counts), regex=pat, base_regex=pat_a,
                new_merges=len(new_merges), new_tokens=len(new_tokens), first_new_id=top + 1,
                secs=round(time.time() - t0, 1))
    json.dump(meta, open(out / "meta.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:7])
