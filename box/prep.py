"""
prep.py: parquet -> NFC text, exact-dedup, FLORES-decontaminated, with held-out.

Decontamination drops any document sharing a whitespace 10-gram with FLORES-200
dev or devtest of the document's language or English (Belebele and SIB-200 are
built on FLORES passages, so they are covered too). Counts are reported, because
a pass that removes nothing is evidence the check is broken.

Held-out text is filled FIRST from the random-order stream, so the training
file provably never contains it.

usage: python prep.py <lang_code> <budget_bytes> [heldout_bytes]
"""
import glob, hashlib, json, sys, unicodedata
from pathlib import Path
import pyarrow.parquet as pq

H = Path("/home/ntt")
FL = H / "flores200_dataset"
NGRAM = 10


def flores_grams(codes):
    grams, n = set(), 0
    for split in ("dev", "devtest"):
        for code in codes:
            p = FL / split / f"{code}.{split}"
            for line in p.read_text(encoding="utf-8").splitlines():
                w = unicodedata.normalize("NFC", line).split()
                n += 1
                for i in range(len(w) - NGRAM + 1):
                    grams.add(hash(" ".join(w[i:i + NGRAM])))
    return grams, n


def contaminated(text, grams):
    w = text.split()
    return any(hash(" ".join(w[i:i + NGRAM])) in grams for i in range(len(w) - NGRAM + 1))


def files_for(code):
    if code == "eng_Latn":
        return sorted(glob.glob(str(H / "raw/fwedu/sample/10BT/*.parquet")))
    return sorted(glob.glob(str(H / f"raw/fw2/data/{code}/train/*.parquet")))


def main(code, budget, held_bytes=0):
    grams, nsent = flores_grams(sorted({code, "eng_Latn"}))
    out_p = H / "corpus" / f"{code}.jsonl"
    held_p = H / "corpus" / f"{code}.heldout.jsonl"
    st = dict(docs=0, short=0, dup=0, contam=0, kept=0, bytes=0, held=0)
    seen = set()
    out = open(out_p, "w", encoding="utf-8")
    held = open(held_p, "w", encoding="utf-8") if held_bytes else None
    done = False
    for f in files_for(code):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=4000, columns=["text"]):
            for v in batch.column(0):
                t = v.as_py()
                st["docs"] += 1
                if not t or len(t) < 200:
                    st["short"] += 1
                    continue
                t = unicodedata.normalize("NFC", t).strip()
                h = hashlib.blake2b(t.encode(), digest_size=8).digest()
                if h in seen:
                    st["dup"] += 1
                    continue
                seen.add(h)
                if contaminated(t, grams):
                    st["contam"] += 1
                    continue
                b = len(t.encode())
                if held is not None and st["held"] < held_bytes:
                    held.write(json.dumps(t, ensure_ascii=False) + "\n")
                    st["held"] += b
                    continue
                out.write(json.dumps(t, ensure_ascii=False) + "\n")
                st["bytes"] += b
                st["kept"] += 1
                if st["bytes"] >= budget:
                    done = True
                    break
            if done:
                break
        if done:
            break
    out.close()
    held and held.close()
    st.update(code=code, flores_sentences_indexed=nsent, budget=budget)
    json.dump(st, open(H / "logs" / f"prep.{code}.json", "w"), indent=1)
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(float(sys.argv[2])), int(float(sys.argv[3])) if len(sys.argv) > 3 else 0)
