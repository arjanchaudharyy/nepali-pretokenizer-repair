"""Rebuild data/ne_test_sample.txt: the FineWeb-2 npi_Deva test split (ODC-By), one document per line,
with each document's internal newlines replaced by a space. Used by tokenizer_extras.py and
results/revision/devanagari_only.py.  Run from the repo root: python fetch_ne_test_sample.py
"""
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

f = hf_hub_download("HuggingFaceFW/fineweb-2", "data/npi_Deva/test/000_00000.parquet", repo_type="dataset")
texts = pq.read_table(f, columns=["text"]).column("text").to_pylist()
out = Path("data/ne_test_sample.txt")
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(t.replace("\n", " ") for t in texts) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(texts)} documents, {out.stat().st_size:,} bytes)")
