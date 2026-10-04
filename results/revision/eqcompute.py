"""
eqcompute.py (review R2 W4): equal-compute comparison stated honestly, plus wall-clock.

Equal-compute checkpoints: results/cpt/evals/<m>_R{0,1}_s0_eqcompute.per.json, i.e. the
R0 and R1 runs at R2's step count (545 Llama / 556 Qwen), cut mid-cosine (NOT annealed),
scored under the registered (BOS) prefix on the first ~1 MB of the held-out slice only
(567 Nepali / 620 English pieces). They are compared with the FINAL R2 checkpoint
(<m>_R2_s0.per.json, same registered prefix) restricted to the same first pieces
(asserted byte-identical).
Reported: X@R2-steps minus final R2 (negative = the equal-compute X checkpoint is better
than final R2), byte-weighted, with piece-level and document-clustered paired bootstrap
(B=10,000, seed 0, doc heuristic of cluster_bootstrap.py, doc ids taken from the full
piece list and truncated to the first pieces).
No EOS-prefix scores exist for the eq-compute checkpoints (eval_eos.sh did not rescore them).

Wall-clock: runs/<tag>/summary.json "secs" (training only; evaluations of earlier arms ran
concurrently on the same GPUs, see fast_all.sh, so these are not clean timings).

Run from the repo root:  .venv/bin/python results/revision/eqcompute.py
Output: results/revision/eqcompute.json
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cluster_bootstrap import ROOT, R, KEYS, doc_ids, compare  # noqa: E402

out = dict(eqcompute={}, wallclock={})
for m in ("llama", "qwen"):
    fin = json.load(open(R / "evals" / f"{m}_R2_s0.per.json"))
    summ = {a: json.load(open(R / "runs" / f"{m}_{a}_s0" / "summary.json")) for a in ("R0", "R1", "R2")}
    for a in ("R0", "R1"):
        eq = json.load(open(R / "evals" / f"{m}_{a}_s0_eqcompute.per.json"))
        for lang, k in KEYS.items():
            n = len(eq[k]["bytes"])
            r2 = {kk: v[:n] for kk, v in fin[k].items()}
            docs = doc_ids(fin[k]["bytes"])[:n]
            docs = np.unique(docs, return_inverse=True)[1]
            d = compare(eq[k], r2, docs)
            # the full-run X - R2 on the same pieces, for reference
            full = json.load(open(R / "evals" / f"{m}_{a}_s0.per.json"))
            fx = {kk: v[:n] for kk, v in full[k].items()}
            d["final_X_minus_final_R2_same_pieces"] = compare(fx, r2, docs)["diff"]
            ea = json.load(open(R / "evals" / f"{m}_{a}_s0_eqcompute.json"))
            d["eq_checkpoint"] = ea["model"]
            d["eq_bpb"] = float(np.sum(eq[k]["nll_bits"]) / np.sum(eq[k]["bytes"]))
            d["finalR2_bpb_same_pieces"] = float(np.sum(r2["nll_bits"]) / np.sum(r2["bytes"]))
            out["eqcompute"][f"{m}|{lang}|{a}@R2steps-R2final"] = d
    s = {a: summ[a]["secs"] for a in summ}
    out["wallclock"][m] = dict(secs=s, steps={a: summ[a]["steps"] for a in summ},
                               tokens={a: summ[a]["tokens"] for a in summ},
                               flops={a: summ[a]["train_flops"] for a in summ},
                               R2_over_R1_secs=s["R2"] / s["R1"], R2_over_R0_secs=s["R2"] / s["R0"],
                               R2_over_R1_flops=summ["R2"]["train_flops"] / summ["R1"]["train_flops"],
                               R2_over_R0_flops=summ["R2"]["train_flops"] / summ["R0"]["train_flops"],
                               R2_over_R1_tokens=summ["R2"]["tokens"] / summ["R1"]["tokens"],
                               R2_over_R0_tokens=summ["R2"]["tokens"] / summ["R0"]["tokens"])

json.dump(out, open(HERE / "eqcompute.json", "w"), indent=1)
for k, d in out["eqcompute"].items():
    print(f"{k:28} {d['diff']:+.4f} piece[{d['piece_lo']:+.4f},{d['piece_hi']:+.4f}] "
          f"cluster[{d['cluster_lo']:+.4f},{d['cluster_hi']:+.4f}] pieces={d['pieces']} docs={d['docs']} "
          f"eq={d['eq_bpb']:.4f} R2={d['finalR2_bpb_same_pieces']:.4f} (finalX-R2 same pieces {d['final_X_minus_final_R2_same_pieces']:+.4f})")
for m, w in out["wallclock"].items():
    print(m, w["secs"], {k: round(v, 3) for k, v in w.items() if k.startswith("R2_")})
