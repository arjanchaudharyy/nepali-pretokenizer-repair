# Amendment 5: matched-steps, second seed, two-tokenization scoring (written before running)

Written 2026-10-05, after all earlier results and two external reviews. Motivation: at equal bytes R2
takes 26% fewer optimiser steps than R1, and on Qwen the R1 checkpoint saved at R2's final step is
level with final R2 (-0.0000 [-0.0021, +0.0019]). The R2 - R1 gap may therefore be training length.
Canonical-tokenization BPB may also understate the extended arms.

## Runs (nothing else changes: same documents, K=32,000, lr 1e-4, global batch 256, micro 4, 4x H200)

1. **Llama R2, matched steps**: the BOS-consistent setting (`data/llamabos_R2`), seed 0, trained for
   737 steps (R1's count) instead of 546. Steps past one epoch take windows from a second permutation
   (seed + 1000), so 35% of the stream is seen twice. The cosine schedule spans all 737 steps.
2. **Qwen R2, matched steps**: `data/qwen_R2`, seed 0, 748 steps (R1's count) instead of 556; same rule.
3. **Qwen R1 and R2, seed 1**: the main runs repeated with seed 1 (data order only; initialisation is
   deterministic).

## Scoring

Registered protocol: byte-matched held-out Nepali BPB, pieces of ~2,000 bytes, after BOS (Llama BOS
setting) or end-of-text (Qwen), document-clustered paired bootstrap (B = 10,000), claim rules applied
in order (better / non-inferior within +0.01 / worse / no detectable difference).

**Two-tokenization BPB** (new, all R0/R1/R2 final models of the Llama BOS rerun and the Qwen main runs,
and the new runs): for each piece, -log2( p(canonical tokenization) + p(base-tokenizer tokenization) ) /
bytes, both scored by the same model after the same prefix. Every base token is in the extended
vocabularies, so this is a valid lower bound on the marginal likelihood that is at least as tight as the
canonical score. For R0 the two tokenizations coincide and the score is unchanged.

## Predictions

- P5a (Qwen): matched-steps R2 - R1 is non-inferior (95% interval entirely below +0.01).
- P5b (Llama): matched-steps R2 - R1 is smaller than +0.0256 but still worse (interval entirely above 0).
- P5c: R0 remains better than matched-steps R2 on both models.
- P5d: on Qwen, R2 - R1 with seed 1 is within 0.005 of the seed-0 value (+0.0094).
- P5e: two-tokenization scoring lowers R1 and R2 BPB by less than 0.005 and changes no verdict.

## Interpretation fixed in advance

If matched-steps R2 is non-inferior to R1, we report that the R2 - R1 gap at equal bytes is
explained by training length and that the repair gives the same quality at matched steps. If it
remains worse, we report the gap as a cost of the repair under this recipe. Either way, every run is
reported.
