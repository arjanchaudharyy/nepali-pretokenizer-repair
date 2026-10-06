# Amendment 8: second seed for equal-bytes Llama R2 (written before running)

Written 2026-10-06 after an external review noted that equal-bytes Llama R2 (BOS setting) is the only
reported arm with a single seed.

Run: Llama R2, BOS setting (`data/llamabos_R2`), seed 1, equal bytes (546 steps), learning rate 1e-4,
otherwise identical to `bos_llama_R2_s0`.

Prediction: P8a. Averaged over two seeds, equal-bytes Llama R2 - R1 stays worse (interval entirely above 0)
and within 0.005 of the seed-0 value (+0.0256).

Analysis as in amendment 6 (seed-averaged per-piece bits, document-clustered bootstrap).
