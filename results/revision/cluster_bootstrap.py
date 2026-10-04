"""
cluster_bootstrap.py (review R2 W7): document-clustered paired bootstrap for every
CPT comparison, next to the original piece-level bootstrap.

Document ids are not stored with the per-piece outputs. They are reconstructed with
the reviewer's heuristic: pieces are in document order (box/eval.py bpb_chunks splits
each document at whitespace into pieces of <= 2,000 bytes), so a piece of < 1,900
bytes is taken to END a document; the final piece of the list closes the last
document. A document whose last piece happens to be >= 1,900 bytes is merged with the
following one (makes clusters larger, i.e. conservative).

Statistic (identical to analyze_cpt.py): byte-weighted difference in bits per byte,
sum(nll_a - nll_b) / sum(bytes) over the resample. B = 10,000, seed 0.
  piece-level: resample pieces (reproduces results/cpt_analysis*.json exactly)
  clustered:   resample documents with replacement, keep all pieces of each document

Run from the repo root:  .venv/bin/python results/revision/cluster_bootstrap.py
Output: results/revision/cluster_bootstrap.json
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/cpt"
B, SEED, CUT = 10_000, 0, 1900
KEYS = {"ne": "npi_Deva.heldout.jsonl", "en": "eng_Latn.heldout.jsonl"}
PREFIXES = {"registered_BOS": "evals", "training_separator_EOS": "evals_eos"}
EXPS = {"A_llama": lambda a: f"llama_{a}_s0", "A_qwen": lambda a: f"qwen_{a}_s0",
        "B1_llama": lambda a: f"B1_llama_{a}", "BOS_llama": lambda a: f"bos_llama_{a}_s0"}
PAIRS = (("R1", "R0"), ("R2", "R0"), ("R2", "R1"))


def doc_ids(byts, cut=CUT):
    ids, d = [], 0
    for b in byts:
        ids.append(d)
        if b < cut:
            d += 1
    return np.array(ids)


def piece_boot(na, nb, by):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(by), size=(B, len(by)))
    d = (na[idx].sum(1) - nb[idx].sum(1)) / by[idx].sum(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def cluster_boot(na, nb, by, docs):
    D = docs.max() + 1
    dd = np.bincount(docs, weights=na - nb, minlength=D)
    db = np.bincount(docs, weights=by, minlength=D)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, D, size=(B, D))
    d = dd[idx].sum(1) / db[idx].sum(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def compare(pa, pb, docs=None):
    a, b = pa, pb
    assert a["bytes"] == b["bytes"], "pieces differ between arms"
    na, nb, by = (np.array(x, dtype=float) for x in (a["nll_bits"], b["nll_bits"], a["bytes"]))
    docs = doc_ids(a["bytes"]) if docs is None else docs
    plo, phi = piece_boot(na, nb, by)
    clo, chi = cluster_boot(na, nb, by, docs)
    return dict(diff=float((na.sum() - nb.sum()) / by.sum()), piece_lo=plo, piece_hi=phi,
                cluster_lo=clo, cluster_hi=chi, width_ratio=(chi - clo) / (phi - plo),
                pieces=int(len(by)), docs=int(docs.max() + 1))


def verdict(lo, hi, margin=0.01):
    if hi < 0:
        return "better"
    if hi < margin:
        return "non-inferior"
    return "no detectable difference" if lo < 0 else "worse"


def load_per(evdir, tag):
    p = R / evdir / f"{tag}.per.json"
    return json.load(open(p)) if p.exists() else None


def main():
    out = dict(B=B, seed=SEED, heuristic=f"piece < {CUT} bytes ends a document", docs={}, tests={})
    ref = load_per("evals", "llama_R0_s0")
    for lang, k in KEYS.items():
        by = np.array(ref[k]["bytes"])
        docs = doc_ids(by)
        sizes = np.bincount(docs)
        out["docs"][lang] = dict(pieces=int(len(by)), docs=int(docs.max() + 1),
                                 pieces_per_doc_mean=float(sizes.mean()), pieces_per_doc_max=int(sizes.max()),
                                 last_piece_bytes=int(by[-1]), median_piece_bytes=float(np.median(by)))
    for pname, evdir in PREFIXES.items():
        for ename, tagf in EXPS.items():
            per = {a: load_per(evdir, tagf(a)) for a in ("R0", "R1", "R2")}
            for lang, k in KEYS.items():
                for x, y in PAIRS:
                    if not (per[x] and per[y]):
                        continue
                    d = compare(per[x][k], per[y][k])
                    d["verdict_piece"] = verdict(d["piece_lo"], d["piece_hi"])
                    d["verdict_cluster"] = verdict(d["cluster_lo"], d["cluster_hi"])
                    out["tests"][f"{pname}|{ename}|{lang}|{x}-{y}"] = d
    json.dump(out, open(ROOT / "results/revision/cluster_bootstrap.json", "w"), indent=1)
    print(json.dumps(out["docs"], indent=1))
    print(f"{'prefix|exp|lang|pair':48} {'diff':>8}  {'piece 95% CI':>20}  {'doc-cluster 95% CI':>20}  w-ratio  verdict(cluster)")
    for k, d in out["tests"].items():
        print(f"{k:48} {d['diff']:+.4f}  [{d['piece_lo']:+.4f},{d['piece_hi']:+.4f}]  "
              f"[{d['cluster_lo']:+.4f},{d['cluster_hi']:+.4f}]  {d['width_ratio']:.2f}   {d['verdict_cluster']}")


if __name__ == "__main__":
    main()
