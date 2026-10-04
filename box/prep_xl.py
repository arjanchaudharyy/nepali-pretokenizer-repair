"""
prep_xl.py: local, CPU-only variant of prep.py for the cross-lingual sweep.
FineWeb-2 TEST split parquet -> NFC text, exact-dedup, FLORES-decontaminated
(any doc sharing a whitespace 10-gram with FLORES dev/devtest of the language
or English is dropped). Uses ALL surviving text (no budget, no held-out).

usage: python box/prep_xl.py <lang_code>
"""
import glob, hashlib, json, sys, unicodedata
from pathlib import Path
import pyarrow.parquet as pq

H = Path(__file__).resolve().parent.parent
FL = H / "flores200_dataset"
NGRAM = 10


def flores_grams(codes):
    grams, n = set(), 0
    for split in ("dev", "devtest"):
        for code in codes:
            for line in (FL / split / f"{code}.{split}").read_text(encoding="utf-8").splitlines():
                w = unicodedata.normalize("NFC", line).split()
                n += 1
                for i in range(len(w) - NGRAM + 1):
                    grams.add(hash(" ".join(w[i:i + NGRAM])))
    return grams, n


def contaminated(text, grams):
    w = text.split()
    return any(hash(" ".join(w[i:i + NGRAM])) in grams for i in range(len(w) - NGRAM + 1))


def main(code):
    grams, nsent = flores_grams(sorted({code, "eng_Latn"}))
    files = sorted(glob.glob(str(H / f"data/xl/raw/data/{code}/test/*.parquet")))
    out_p = H / "data/xl" / f"{code}.txt"
    st = dict(docs=0, short=0, dup=0, contam=0, kept=0, bytes=0)
    seen = set()
    with open(out_p, "w", encoding="utf-8") as out:
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=4000, columns=["text"]):
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
                    out.write(t + "\n")
                    st["bytes"] += len(t.encode()) + 1
                    st["kept"] += 1
    st.update(code=code, flores_sentences_indexed=nsent, parquet_bytes=sum(Path(f).stat().st_size for f in files))
    json.dump(st, open(H / "data/xl" / f"prep.{code}.json", "w"), indent=1)
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
