"""
amendment5.py: the comparisons registered in PREREG_AMENDMENT5.md, with the same statistic, document
reconstruction and claim rules as cluster_bootstrap.py / analyze_cpt.py.

Run from the repo root after sync:  .venv/bin/python results/revision/amendment5.py
Inputs: results/cpt/evals/*.per.json (registered prefix) and results/cpt/evals_marg/*.per.json.
Output: results/revision/amendment5.json
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_bootstrap import doc_ids, piece_boot, cluster_boot  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EV, MG = ROOT / "results/cpt/evals", ROOT / "results/cpt/evals_marg"
NE = "npi_Deva.heldout.jsonl"


def verdict(lo, hi, margin=0.01):
    if hi < 0:
        return "better"
    if hi < margin:
        return "non-inferior"
    return "worse" if lo > 0 else "no detectable difference"


def per(name, marg=False, canon=False):
    if marg:
        d = json.load(open(MG / f"{name}.per.json"))
        return np.array(d["nll_bits_canon" if canon else "nll_bits"]), np.array(d["bytes"])
    d = json.load(open(EV / f"{name}.per.json"))[NE]
    return np.array(d["nll_bits"]), np.array(d["bytes"])


def cmp(a, b, marg=False):
    (na, ba), (nb, bb) = per(a, marg), per(b, marg)
    assert (ba == bb).all(), (a, b)
    docs = doc_ids(ba)
    diff = float((na.sum() - nb.sum()) / ba.sum())
    plo, phi = piece_boot(na, nb, ba)
    clo, chi = cluster_boot(na, nb, ba, docs)
    return dict(a=a, b=b, diff=diff, piece=[plo, phi], cluster=[clo, chi], verdict=verdict(clo, chi),
                bpb_a=float(na.sum() / ba.sum()), bpb_b=float(nb.sum() / bb.sum()))


out = {}
CMP = {
    "qwen_msR2_minus_R1": ("ms_qwen_R2_s0", "qwen_R1_s0"),
    "qwen_msR2_minus_R0": ("ms_qwen_R2_s0", "qwen_R0_s0"),
    "qwen_msR2_minus_R2": ("ms_qwen_R2_s0", "qwen_R2_s0"),
    "llama_msR2_minus_R1": ("ms_bos_llama_R2_s0", "bos_llama_R1_s0"),
    "llama_msR2_minus_R0": ("ms_bos_llama_R2_s0", "bos_llama_R0_s0"),
    "llama_msR2_minus_R2": ("ms_bos_llama_R2_s0", "bos_llama_R2_s0"),
    "qwen_s1_R2_minus_R1": ("qwen_R2_s1", "qwen_R1_s1"),
    "qwen_s0_R2_minus_R1": ("qwen_R2_s0", "qwen_R1_s0"),
    "qwen_R1_s1_minus_s0": ("qwen_R1_s1", "qwen_R1_s0"),
    "qwen_R2_s1_minus_s0": ("qwen_R2_s1", "qwen_R2_s0"),
}
for k, (a, b) in CMP.items():
    try:
        out[k] = cmp(a, b)
    except FileNotFoundError as e:
        out[k] = dict(missing=str(e))
MCMP = {"qwen_two_R2_minus_R1": ("qwen_R2_s0", "qwen_R1_s0"), "qwen_two_R1_minus_R0": ("qwen_R1_s0", "qwen_R0_s0"),
        "qwen_two_R2_minus_R0": ("qwen_R2_s0", "qwen_R0_s0"),
        "llama_two_R2_minus_R1": ("bos_llama_R2_s0", "bos_llama_R1_s0"), "llama_two_R1_minus_R0": ("bos_llama_R1_s0", "bos_llama_R0_s0"),
        "llama_two_R2_minus_R0": ("bos_llama_R2_s0", "bos_llama_R0_s0")}
for k, (a, b) in MCMP.items():
    try:
        out[k] = cmp(a, b, marg=True)
    except FileNotFoundError as e:
        out[k] = dict(missing=str(e))
for f in sorted(MG.glob("*[0-9].json")):
    out["marg_summary_" + f.stem] = json.load(open(f))
json.dump(out, open(ROOT / "results/revision/amendment5.json", "w"), indent=1)
for k, v in out.items():
    if "diff" in v:
        print(f"{k:<26} {v['diff']:+.4f} piece[{v['piece'][0]:+.4f},{v['piece'][1]:+.4f}] "
              f"cluster[{v['cluster'][0]:+.4f},{v['cluster'][1]:+.4f}] {v['verdict']}  ({v['bpb_a']:.4f} vs {v['bpb_b']:.4f})")
    elif "missing" in v:
        print(f"{k:<26} missing")
    else:
        print(k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items() if x not in ("model", "base")})
