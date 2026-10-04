"""
analyze_cpt.py: the pre-registered analysis (PREREG_AMENDMENT.md) of the CPT runs.

Inputs (copied from the box): results/cpt/evals/*.json, *.per.json; results/cpt/runs/*/summary.json
Outputs: results/cpt_analysis.json, paper/sections/cpt_table.tex, paper/fig_cpt.pdf
"""
import json, glob, os
from pathlib import Path
import numpy as np

R = Path("results/cpt")
B, SEED = 10_000, 0
MODELS = {"llama": "Llama-3.2-1B", "qwen": "Qwen3-0.6B"}
ARMS = ["R0", "R1", "R2"]


def load(tag):
    p = R / "evals" / f"{tag}.json"
    if not p.exists():
        return None, None
    per = R / "evals" / f"{tag}.per.json"
    return json.load(open(p)), (json.load(open(per)) if per.exists() else None)


def paired(per_a, per_b, key):
    """Mean of per-piece BPB difference (a - b), weighting pieces by bytes, with bootstrap CI."""
    a, b = per_a[key], per_b[key]
    assert a["bytes"] == b["bytes"], "pieces differ between arms"
    na, nb, by = map(np.array, (a["nll_bits"], b["nll_bits"], a["bytes"]))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(by), size=(B, len(by)))
    diff = (na[idx].sum(1) - nb[idx].sum(1)) / by[idx].sum(1)
    point = (na.sum() - nb.sum()) / by.sum()
    return dict(diff=float(point), lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)),
                pieces=int(len(by)))


def verdict(d, margin=0.01):
    if d["hi"] < 0:
        return "better"
    if d["hi"] < margin:
        return "non-inferior"
    return "no detectable difference" if d["lo"] < 0 else "worse"


def acc_ci(xs):
    xs = np.array(xs)
    rng = np.random.default_rng(SEED)
    bs = xs[rng.integers(0, len(xs), size=(B, len(xs)))].mean(1)
    return float(xs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


out = {}
for m, name in MODELS.items():
    res = {}
    for tag in ["base"] + [f"{a}_s0" for a in ARMS]:
        r, per = load(f"{m}_{tag}")
        if r is None:
            continue
        s = R / "runs" / f"{m}_{tag}" / "summary.json"
        if s.exists():
            r["train"] = json.load(open(s))
        if per:
            for k in ("belebele_npi_Deva", "belebele_eng_Latn"):
                if k in per:
                    r[k + "_ci"] = acc_ci(per[k])
        res[tag] = dict(r=r, per=per)
    tests = {}
    for lang, key in (("ne", "npi_Deva.heldout.jsonl"), ("en", "eng_Latn.heldout.jsonl")):
        for a, b in (("R2_s0", "R0_s0"), ("R2_s0", "R1_s0"), ("R1_s0", "R0_s0")):
            if a in res and b in res and res[a]["per"] and res[b]["per"]:
                d = paired(res[a]["per"], res[b]["per"], key)
                d["verdict"] = verdict(d)
                tests[f"{lang}:{a}-{b}"] = d
    eq = {}
    for a in ("R0", "R1"):
        r, _ = load(f"{m}_{a}_s0_eqcompute")
        if r:
            eq[a] = r
    out[m] = dict(name=name, evals={k: v["r"] for k, v in res.items()}, tests=tests, eqcompute=eq)

json.dump(out, open("results/cpt_analysis.json", "w"), indent=1)
for m, d in out.items():
    print(f"\n== {d['name']}")
    for tag, r in d["evals"].items():
        t = r.get("train", {})
        print(f"{tag:6} ne_chunk={r.get('bpb_chunk2k_ne', float('nan')):.4f} en_chunk={r.get('bpb_chunk2k_en', float('nan')):.4f} "
              f"ne_win={r.get('bpb_native_ne', float('nan')):.4f} flores_ne={r.get('bpb_flores_ne', float('nan')):.4f} "
              f"bel_ne={r.get('belebele_ne', float('nan')):.3f} chrf_ne_en={r.get('chrf_ne_en', float('nan')):.1f} "
              f"chrf_en_ne={r.get('chrf_en_ne', float('nan')):.1f} gen_Bps={r.get('gen_bytes_per_s', float('nan')):.0f} "
              f"tokens={t.get('tokens', 0)/1e6:.0f}M secs={t.get('secs', 0):.0f}")
    for k, v in d["tests"].items():
        print(f"  {k:16} diff={v['diff']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] -> {v['verdict']}")
    for a, r in d["eqcompute"].items():
        print(f"  eq-compute {a}: ne_chunk={r.get('bpb_chunk2k_ne', float('nan')):.4f} en_chunk={r.get('bpb_chunk2k_en', float('nan')):.4f}")
