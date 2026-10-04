"""
pretok.py: build the CPT token stream for one (model, arm).

Every arm gets the IDENTICAL byte stream: the first `ne_bytes` of the Nepali
corpus and the first `en_bytes` of the English corpus, split into documents and
interleaved in one fixed order (seed 0). Only the tokenizer differs. Each
document is followed by the tokenizer's end-of-text id.

Output: <out>/train.u32 (uint32 token ids), <out>/meta.json.

usage: python pretok.py <tokenizer.json> <eos_token> <ne_bytes> <en_bytes> <out>
"""
import json, random, sys, time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

H = Path("/home/ntt")
import os
CORPUS = Path(os.environ.get("CORPUS_DIR", "/home/ntt/corpus"))


def docs(path, budget):
    out, got = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(d)
            got += len(d.encode())
            if got >= budget:
                break
    return out, got


def main(tok_path, eos, ne_bytes, en_bytes, out, bos=None):
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer.from_file(tok_path)
    eos_id = tk.token_to_id(eos)
    assert eos_id is not None, eos
    bos_id = tk.token_to_id(bos) if bos else None
    assert bos is None or bos_id is not None, bos
    ne, gne = docs(CORPUS / "npi_Deva.jsonl", int(float(ne_bytes)))
    en, gen = docs(CORPUS / "eng_Latn.jsonl", int(float(en_bytes)))
    order = [(0, i) for i in range(len(ne))] + [(1, i) for i in range(len(en))]
    random.Random(0).shuffle(order)
    src = (ne, en)
    seq = [src[s][i] for s, i in order]
    lang = np.array([s for s, _ in order], dtype=np.uint8)
    ntok = {0: 0, 1: 0}
    with open(out / "train.u32", "wb") as f:
        B = 20000
        for k in range(0, len(seq), B):
            enc = tk.encode_batch(seq[k:k + B], add_special_tokens=False)
            buf = []
            for j, e in enumerate(enc):
                if bos_id is not None:
                    buf.append(bos_id)
                buf.extend(e.ids)
                buf.append(eos_id)
                ntok[int(lang[k + j])] += len(e.ids) + 1 + (bos_id is not None)
            np.asarray(buf, dtype=np.uint32).tofile(f)
    meta = dict(tokenizer=tok_path, eos_id=eos_id, ne_docs=len(ne), en_docs=len(en), ne_bytes=gne,
                en_bytes=gen, ne_tokens=ntok[0], en_tokens=ntok[1], total_tokens=ntok[0] + ntok[1],
                vocab=tk.get_vocab_size(), bos_id=bos_id, secs=round(time.time() - t0, 1))
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:7])
