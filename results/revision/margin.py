"""
margin.py (review R2 W7): what the pre-registered 0.01 BPB non-inferiority margin is,
relative to base-model and R0 held-out Nepali BPB (bpb_chunk2k_ne), under both prefixes.
Run from the repo root:  .venv/bin/python results/revision/margin.py
Output: results/revision/margin.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M = 0.01
out = {}
for evdir, pname in (("evals", "registered_BOS"), ("evals_eos", "training_separator_EOS")):
    for m in ("llama", "qwen"):
        for tag in (f"{m}_base", f"{m}_R0_s0", f"{m}_R1_s0", f"{m}_R2_s0"):
            r = json.load(open(ROOT / "results/cpt" / evdir / f"{tag}.json"))
            b = r["bpb_chunk2k_ne"]
            out[f"{pname}|{tag}"] = dict(bpb_ne=b, margin_pct=100 * M / b)
json.dump(out, open(ROOT / "results/revision/margin.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:40} BPB={v['bpb_ne']:.4f}  0.01 = {v['margin_pct']:.2f}%")
