"""
marg.py: two-tokenization BPB (amendment 5).

For each held-out Nepali piece (the same ~2,000-byte pieces as eval.py's bpb_chunks), score the
model's canonical tokenization and the base model's tokenization with the same model after the same
prefix, and report -log2(p_canonical + p_base) / bytes. Every base token id is unchanged in the
extended vocabularies, so both are valid token sequences for the text and the sum is a lower bound
on the marginal likelihood that is at least as tight as the canonical score.

usage: python marg.py <model_dir> <base_tokenizer_dir> <out.json>
"""
import json, math, sys
import numpy as np
import torch
from transformers import AutoTokenizer
import eval as E

LN2 = math.log(2)
mdir, bdir, out = sys.argv[1:4]
m = E.M(mdir)
base = AutoTokenizer.from_pretrained(bdir)
V = m.model.get_input_embeddings().weight.shape[0]

pieces, nbytes = [], 0
with open(E.H / "corpus/npi_Deva.heldout.jsonl", encoding="utf-8") as f:  # identical to bpb_chunks
    for line in f:
        d = E.nfc(json.loads(line))
        cur = ""
        for w in d.split(" "):
            cand = (cur + " " + w) if cur else w
            if len(cand.encode()) > 2000 and cur:
                pieces.append(cur); cur = w
            else:
                cur = cand
        if cur:
            pieces.append(cur)
        nbytes += len(d.encode())
        if nbytes >= 4e6:
            break

canon = [m.enc(x) for x in pieces]
alt = [base(x, add_special_tokens=False)["input_ids"] for x in pieces]
assert all(max(a) < V for a in alt)
assert all(base.decode(a) == x for a, x in zip(alt[:200], pieces[:200]))
same = [c == a for c, a in zip(canon, alt)]
nc = np.array(m.batch_nll([[m.sep] + c for c in canon], bs=16))
na = np.array([c for c in nc]) if all(same) else np.array(m.batch_nll([[m.sep] + a for a in alt], bs=16))
na = np.where(same, np.inf, na)  # identical sequences are one tokenization, counted once
by = np.array([len(x.encode()) for x in pieces])
two = -np.logaddexp(-nc, -na)  # nats
r = dict(model=mdir, base=bdir, pieces=len(pieces), identical=int(sum(same)),
         bpb_canonical=float(nc.sum() / LN2 / by.sum()), bpb_two=float(two.sum() / LN2 / by.sum()),
         bpb_base_tok=float(np.where(same, nc, na).sum() / LN2 / by.sum()),
         share_base_wins=float(np.mean(na < nc)))
json.dump(r, open(out, "w"), indent=1)
json.dump(dict(nll_bits=(two / LN2).tolist(), nll_bits_canon=(nc / LN2).tolist(), bytes=by.tolist()),
          open(out.replace(".json", ".per.json"), "w"))
print(json.dumps(r), flush=True)
