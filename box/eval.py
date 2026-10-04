"""
eval.py: tokenizer-independent evaluation of one checkpoint.

Every number is normalised by UTF-8 BYTES, never by tokens, because the arms
use different tokenizers and per-token quantities are not comparable.

  bpb_native_{ne,en}   held-out native web text (never trained on); each doc is
                       prefixed by the end-of-text id the model saw between CPT
                       docs; long docs scored with a strided window (context
                       2048, stride 1024) so window edges cost every arm alike.
  bpb_flores_{ne,en}   FLORES-200 devtest, sentence by sentence.
  belebele_{ne,en}     zero-shot, 4 options, scored on the ANSWER TEXT by
                       log-likelihood per byte (letter-choice prompts put small
                       models at chance in both languages; Arkios 2608.30092).
  chrf_{ne_en,en_ne}   FLORES devtest, 5-shot from FLORES dev, greedy, chrF++.

usage: python eval.py <model_dir> <out.json> [--quick]
"""
import json, math, sys, time, unicodedata
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

H = Path("/home/ntt")
FL = H / "flores200_dataset"
dev = torch.device("cuda")
LN2 = math.log(2)


def nfc(s):
    return unicodedata.normalize("NFC", s)


def flores(code, split="devtest"):
    return [nfc(x) for x in (FL / split / f"{code}.{split}").read_text(encoding="utf-8").splitlines()]


class M:
    def __init__(self, d):
        self.tok = AutoTokenizer.from_pretrained(d)
        self.model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16,
                                                          attn_implementation="sdpa").to(dev).eval()
        # document prefix: the model's BOS if it has one (Llama), else end-of-text (Qwen).
        # Same rule for every arm of a model, so arm comparisons are unaffected by it.
        self.sep = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id

    def enc(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def nll_tokens(self, ids, ctx=2048, stride=1024):
        """Summed NLL (nats) of ids[1:] given ids[0] (the separator)."""
        total, n = 0.0, len(ids)
        start = 0
        prev_end = 1
        while True:
            end = min(start + ctx, n)
            x = torch.tensor([ids[start:end]], device=dev)
            logits = self.model(input_ids=x).logits[0].float()
            lp = torch.log_softmax(logits[:-1], -1)
            tgt = x[0, 1:]
            nll = -lp.gather(1, tgt[:, None])[:, 0]
            # score only targets not already scored: positions >= prev_end
            first = max(prev_end - start - 1, 0)
            total += nll[first:].sum().item()
            prev_end = end
            if end == n:
                break
            start += stride
        return total

    @torch.no_grad()
    def batch_nll(self, seqs, bs=32):
        """Summed NLL of each seq[1:] (short sequences; padded batch)."""
        out = []
        for k in range(0, len(seqs), bs):
            chunk = seqs[k:k + bs]
            L = max(len(s) for s in chunk)
            x = torch.full((len(chunk), L), self.sep, device=dev)
            m = torch.zeros((len(chunk), L), device=dev)
            for i, s in enumerate(chunk):
                x[i, :len(s)] = torch.tensor(s, device=dev)
                m[i, :len(s)] = 1
            logits = self.model(input_ids=x, attention_mask=m).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1)
            nll = -lp.gather(2, x[:, 1:, None])[..., 0] * m[:, 1:]
            out.extend(nll.sum(1).tolist())
        return out


def bpb_docs(m, path, max_bytes):
    nats, nbytes = 0.0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = nfc(json.loads(line))
            if not d:
                continue
            nats += m.nll_tokens([m.sep] + m.enc(d))
            nbytes += len(d.encode())
            if nbytes >= max_bytes:
                break
    return nats / LN2 / nbytes, nbytes


def bpb_chunks(m, path, max_bytes, chunk=2000):
    """Byte-matched context: split each held-out document into ~`chunk`-byte pieces
    at whitespace and score every piece independently, so every arm conditions on
    the same bytes regardless of how many tokens they take."""
    pieces, nbytes = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = nfc(json.loads(line))
            cur = ""
            for w in d.split(" "):
                cand = (cur + " " + w) if cur else w
                if len(cand.encode()) > chunk and cur:
                    pieces.append(cur); cur = w
                else:
                    cur = cand
            if cur:
                pieces.append(cur)
            nbytes += len(d.encode())
            if nbytes >= max_bytes:
                break
    nll = m.batch_nll([[m.sep] + m.enc(x) for x in pieces], bs=16)
    return sum(nll) / LN2 / sum(len(x.encode()) for x in pieces)


def bpb_flores(m, code):
    sents = flores(code)
    nll = m.batch_nll([[m.sep] + m.enc(s) for s in sents])
    return sum(nll) / LN2 / sum(len(s.encode()) for s in sents)


def belebele(m, code):
    from datasets import load_dataset
    ds = load_dataset("facebook/belebele", code, split="test")
    correct = 0
    for ex in ds:
        prompt = f"{ex['flores_passage']}\n{ex['question']}\n"
        p_ids = [m.sep] + m.enc(prompt)
        opts = [ex[f"mc_answer{i}"] for i in range(1, 5)]
        seqs = [p_ids + m.enc(o) for o in opts]
        full = m.batch_nll(seqs, bs=4)
        base = m.batch_nll([p_ids], bs=1)[0]
        scores = [-(f - base) / max(1, len(o.encode())) for f, o in zip(full, opts)]
        pred = max(range(4), key=lambda i: scores[i])
        correct += int(pred + 1 == int(ex["correct_answer_num"]))
    return correct / len(ds)


@torch.no_grad()
def chrf(m, src_code, tgt_code, n=1012, shots=5):
    import sacrebleu
    ds, dt = flores(src_code, "dev"), flores(tgt_code, "dev")
    ss, tt = flores(src_code)[:n], flores(tgt_code)[:n]
    names = {"npi_Deva": "Nepali", "eng_Latn": "English"}
    head = "".join(f"{names[src_code]}: {ds[i]}\n{names[tgt_code]}: {dt[i]}\n\n" for i in range(shots))
    m.tok.padding_side = "left"
    if m.tok.pad_token is None:
        m.tok.pad_token = m.tok.eos_token
    hyps = []
    for k in range(0, len(ss), 32):
        prompts = [head + f"{names[src_code]}: {s}\n{names[tgt_code]}:" for s in ss[k:k + 32]]
        enc = m.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(dev)
        gen = m.model.generate(**enc, max_new_tokens=512, do_sample=False, pad_token_id=m.tok.pad_token_id)
        for g in gen[:, enc["input_ids"].shape[1]:]:
            hyps.append(m.tok.decode(g, skip_special_tokens=True).split("\n")[0].strip())
    return sacrebleu.corpus_chrf(hyps, [tt], word_order=2).score


@torch.no_grad()
def gen_speed(m, n=64, new=128):
    """Greedy continuation of FLORES Nepali sentences: UTF-8 bytes generated per second."""
    m.tok.padding_side = "left"
    if m.tok.pad_token is None:
        m.tok.pad_token = m.tok.eos_token
    prompts = flores("npi_Deva", "dev")[:n]
    enc = m.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(dev)
    m.model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=m.tok.pad_token_id)  # warm-up
    torch.cuda.synchronize(); t = time.time()
    g = m.model.generate(**enc, max_new_tokens=new, min_new_tokens=new, do_sample=False,
                         pad_token_id=m.tok.pad_token_id)
    torch.cuda.synchronize(); dt = time.time() - t
    gen = g[:, enc["input_ids"].shape[1]:]
    nbytes = sum(len(m.tok.decode(x, skip_special_tokens=True).encode()) for x in gen)
    return dict(gen_tokens_per_s=gen.numel() / dt, gen_bytes_per_s=nbytes / dt,
                gen_bytes_per_token=nbytes / gen.numel())


def main(d, out, quick=False):
    t0 = time.time()
    m = M(d)
    r = dict(model=d, vocab=len(m.tok))
    r["bpb_flores_ne"] = bpb_flores(m, "npi_Deva")
    r["bpb_flores_en"] = bpb_flores(m, "eng_Latn")
    mb = 1e6 if quick else 4e6
    r["bpb_native_ne"], r["bytes_native_ne"] = bpb_docs(m, H / "corpus/npi_Deva.heldout.jsonl", mb)
    r["bpb_native_en"], r["bytes_native_en"] = bpb_docs(m, H / "corpus/eng_Latn.heldout.jsonl", mb)
    r["bpb_chunk2k_ne"] = bpb_chunks(m, H / "corpus/npi_Deva.heldout.jsonl", mb)
    r["bpb_chunk2k_en"] = bpb_chunks(m, H / "corpus/eng_Latn.heldout.jsonl", mb)
    ne, en = flores("npi_Deva"), flores("eng_Latn")
    r["ne_tokens_per_byte"] = sum(len(m.enc(s)) for s in ne) / sum(len(s.encode()) for s in ne)
    r["en_tokens_per_byte"] = sum(len(m.enc(s)) for s in en) / sum(len(s.encode()) for s in en)
    if not quick:
        r.update(gen_speed(m))
        r["belebele_ne"] = belebele(m, "npi_Deva")
        r["belebele_en"] = belebele(m, "eng_Latn")
        r["chrf_ne_en"] = chrf(m, "npi_Deva", "eng_Latn")
        r["chrf_en_ne"] = chrf(m, "eng_Latn", "npi_Deva")
    r["secs"] = round(time.time() - t0, 1)
    json.dump(r, open(out, "w"), indent=1)
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--quick" in sys.argv)
