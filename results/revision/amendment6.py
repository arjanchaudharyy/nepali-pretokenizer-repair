"""
amendment6.py: comparisons registered in PREREG_AMENDMENT6.md. Seed-averaged comparisons average the
per-piece bits of each arm's seeds before the document-clustered bootstrap (same statistic and claim
rules as amendment5.py).

Run from the repo root after sync:  .venv/bin/python results/revision/amendment6.py
Output: results/revision/amendment6.json
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_bootstrap import doc_ids, piece_boot, cluster_boot  # noqa: E402
from amendment5 import per, verdict  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def avg(names):
    xs = [per(n) for n in names]
    assert all((b == xs[0][1]).all() for _, b in xs)
    return np.mean([n for n, _ in xs], axis=0), xs[0][1]


def cmp(A, B):
    (na, by), (nb, bb) = avg(A), avg(B)
    assert (by == bb).all()
    docs = doc_ids(by)
    plo, phi = piece_boot(na, nb, by)
    clo, chi = cluster_boot(na, nb, by, docs)
    return dict(a=A, b=B, diff=float((na.sum() - nb.sum()) / by.sum()), piece=[plo, phi], cluster=[clo, chi],
                verdict=verdict(clo, chi), bpb_a=float(na.sum() / by.sum()), bpb_b=float(nb.sum() / by.sum()),
                bpb_a_seeds=[float(per(n)[0].sum() / by.sum()) for n in A],
                bpb_b_seeds=[float(per(n)[0].sum() / by.sum()) for n in B])


Q = dict(R0=["qwen_R0_s0", "qwen_R0_s1"], R1=["qwen_R1_s0", "qwen_R1_s1"], R2=["qwen_R2_s0", "qwen_R2_s1"],
         msR2=["ms_qwen_R2_s0", "ms_qwen_R2_s1"])
L = dict(R0=["bos_llama_R0_s0", "bos_llama_R0_s1"], R1=["bos_llama_R1_s0", "bos_llama_R1_s1"],
         msR2=["ms_bos_llama_R2_s0", "ms_bos_llama_R2_s1"], R2=["bos_llama_R2_s0", "bos_llama_R2_s1"])
CMP = {
    "qwen_2seed_msR2_minus_R1": (Q["msR2"], Q["R1"]), "qwen_2seed_msR2_minus_R0": (Q["msR2"], Q["R0"]),
    "qwen_2seed_R1_minus_R0": (Q["R1"], Q["R0"]), "qwen_2seed_R2_minus_R1": (Q["R2"], Q["R1"]),
    "llama_2seed_msR2_minus_R1": (L["msR2"], L["R1"]), "llama_2seed_msR2_minus_R0": (L["msR2"], L["R0"]),
    "llama_2seed_R1_minus_R0": (L["R1"], L["R0"]),
    # amendment 8
    "llama_2seed_R2_minus_R1": (L["R2"], L["R1"]), "llama_2seed_R2_minus_R0": (L["R2"], L["R0"]),
    "qwen_lr3_msR2_minus_R1": (["lr3_ms_qwen_R2_s0"], ["lr3_qwen_R1_s0"]),
    "qwen_lr3_R1_minus_lr1_R1": (["lr3_qwen_R1_s0"], ["qwen_R1_s0"]),
    "qwen_lr3_msR2_minus_lr1_msR2": (["lr3_ms_qwen_R2_s0"], ["ms_qwen_R2_s0"]),
    "qwen_lr3_R1_minus_R0": (["lr3_qwen_R1_s0"], Q["R0"]),
    "qwen_lr3_msR2_minus_lr1_R0": (["lr3_ms_qwen_R2_s0"], Q["R0"]),
    # amendment 7
    "qwen_lr3_R0_minus_lr1_R0": (["lr3_qwen_R0_s0"], ["qwen_R0_s0"]),
    "qwen_lr3_msR2_minus_lr3_R0": (["lr3_ms_qwen_R2_s0"], ["lr3_qwen_R0_s0"]),
    "qwen_lr3_R1_minus_lr3_R0": (["lr3_qwen_R1_s0"], ["lr3_qwen_R0_s0"]),
}
out = {}
for k, (a, b) in CMP.items():
    try:
        out[k] = cmp(a, b)
    except FileNotFoundError as e:
        out[k] = dict(missing=str(e).split("/")[-1])
json.dump(out, open(ROOT / "results/revision/amendment6.json", "w"), indent=1)
for k, v in out.items():
    if "diff" in v:
        print(f"{k:<30} {v['diff']:+.4f} cluster[{v['cluster'][0]:+.4f},{v['cluster'][1]:+.4f}] {v['verdict']:<13} "
              f"a={['%.4f' % x for x in v['bpb_a_seeds']]} b={['%.4f' % x for x in v['bpb_b_seeds']]}")
    else:
        print(f"{k:<30} missing {v['missing']}")
