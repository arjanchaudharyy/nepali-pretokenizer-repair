"""
analyze_cpt.py: the pre-registered analysis (PREREG_AMENDMENT.md, PREREG_AMENDMENT3.md).

Experiments:
  A   Llama-3.2-1B and Qwen3-0.6B, 1 GB + 1 GB, no warm-up     evals/<m>_<arm>_s0.json
  B1  Llama-3.2-1B, embedding warm-up, 1 GB + 1 GB             evals/B1_<m>_<arm>.json
  B3  Llama-3.2-1B, embedding warm-up, 3 GB + 3 GB             evals/B3_<m>_<arm>.json
Inputs are copied from the box by sync_cpt.sh into results/cpt/.
Output: results/cpt_analysis.json, printed summary.
"""
import json
from pathlib import Path
import numpy as np

R = Path("results/cpt")
B, SEED = 10_000, 0
MODELS = {"llama": "Llama-3.2-1B", "qwen": "Qwen3-0.6B"}
ARMS = ["R0", "R1", "R2"]
EXPS = {"A": lambda m, a: f"{m}_{a}_s0", "B1": lambda m, a: f"B1_{m}_{a}", "B3": lambda m, a: f"B3_{m}_{a}", "BOS": lambda m, a: f"bos_{m}_{a}_s0"}
KEYS = {"ne": "npi_Deva.heldout.jsonl", "en": "eng_Latn.heldout.jsonl"}


EVDIR = "evals"


def load(tag):
    p = R / EVDIR / f"{tag}.json"
    if not p.exists():
        return None, None
    per = R / EVDIR / f"{tag}.per.json"
    r = json.load(open(p))
    s = R / "runs" / tag / "summary.json"
    if s.exists():
        r["train"] = json.load(open(s))
    return r, (json.load(open(per)) if per.exists() else None)


def paired(pa, pb, key):
    a, b = pa[key], pb[key]
    assert a["bytes"] == b["bytes"], "pieces differ between arms"
    na, nb, by = map(np.array, (a["nll_bits"], b["nll_bits"], a["bytes"]))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(by), size=(B, len(by)))
    d = (na[idx].sum(1) - nb[idx].sum(1)) / by[idx].sum(1)
    return dict(diff=float((na.sum() - nb.sum()) / by.sum()), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), pieces=int(len(by)))


def verdict(d, margin=0.01):
    if d["hi"] < 0:
        return "better"
    if d["hi"] < margin:
        return "non-inferior"
    return "no detectable difference" if d["lo"] < 0 else "worse"


def acc_ci(xs):
    xs = np.array(xs, dtype=float)
    rng = np.random.default_rng(SEED)
    bs = xs[rng.integers(0, len(xs), size=(B, len(xs)))].mean(1)
    return [float(xs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


import sys
if len(sys.argv) > 1:
    EVDIR = sys.argv[1]
out = {}
for m, name in MODELS.items():
    base, base_per = load(f"{m}_base")
    out[m] = dict(name=name, base=base, exps={})
    for e, tagf in EXPS.items():
        res, per = {}, {}
        for a in ARMS:
            r, p = load(tagf(m, a))
            if r is None:
                continue
            if p:
                for k in ("belebele_npi_Deva", "belebele_eng_Latn"):
                    if k in p:
                        r[k + "_ci"] = acc_ci(p[k])
            res[a], per[a] = r, p
        if not res:
            continue
        tests = {}
        for lang, key in KEYS.items():
            for x, y in (("R2", "R0"), ("R2", "R1"), ("R1", "R0")):
                if per.get(x) and per.get(y):
                    d = paired(per[x], per[y], key)
                    d["verdict"] = verdict(d)
                    tests[f"{lang}:{x}-{y}"] = d
        eq = {}
        if e == "A":
            for a in ("R0", "R1"):
                r, _ = load(f"{m}_{a}_s0_eqcompute")
                if r:
                    eq[a] = r
        out[m]["exps"][e] = dict(evals=res, tests=tests, eqcompute=eq)

json.dump(out, open("results/cpt_analysis.json" if EVDIR == "evals" else f"results/cpt_analysis_{EVDIR}.json", "w"), indent=1)
f = lambda r, k, fmt=".4f": format(r[k], fmt) if r and k in r else "-"
for m, d in out.items():
    print(f"\n==== {d['name']}")
    b = d["base"]
    if b:
        print(f"base   ne={f(b,'bpb_chunk2k_ne')} en={f(b,'bpb_chunk2k_en')} flores_ne={f(b,'bpb_flores_ne')} "
              f"bel_ne={f(b,'belebele_ne','.3f')} chrf ne>en={f(b,'chrf_ne_en','.1f')} en>ne={f(b,'chrf_en_ne','.1f')} gen_Bps={f(b,'gen_bytes_per_s','.0f')}")
    for e, x in d["exps"].items():
        print(f"-- {e}")
        for a, r in x["evals"].items():
            t = r.get("train", {})
            print(f"{a}  ne={f(r,'bpb_chunk2k_ne')} en={f(r,'bpb_chunk2k_en')} flores_ne={f(r,'bpb_flores_ne')} "
                  f"bel_ne={f(r,'belebele_ne','.3f')} chrf ne>en={f(r,'chrf_ne_en','.1f')} en>ne={f(r,'chrf_en_ne','.1f')} "
                  f"gen_Bps={f(r,'gen_bytes_per_s','.0f')} tokens={t.get('tokens', 0)/1e6:.0f}M")
        for k, v in x["tests"].items():
            print(f"   {k:9} {v['diff']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] {v['verdict']}")
        for a, r in x["eqcompute"].items():
            print(f"   eq-compute {a}: ne={f(r,'bpb_chunk2k_ne')} en={f(r,'bpb_chunk2k_en')}")
