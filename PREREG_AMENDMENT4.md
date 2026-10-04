# Amendment 4: BOS-consistent Llama rerun (written before running)

Written 2026-10-05 after the main and warm-up results and the internal reviews.
Motivation: our continued pretraining never placed Llama's beginning-of-sequence
(BOS) token at document starts, while the registered evaluation prefixes it. The
R2 vs R1 verdict depends on the scoring prefix (+0.006 vs +0.027).

Design: rerun the three Llama-3.2-1B arms (R0, R1, R2) exactly as in the main runs
(same 1 GB + 1 GB documents and order, K = 32,000, lr 1e-4, seed 0, one epoch), except
that every document in the training stream is "BOS document EOS". Evaluate with the
registered protocol (BOS prefix), byte-matched 2,000-byte pieces, document-clustered
paired bootstrap as in the revision. Secondary metrics (Belebele, translation,
generation) are not run, for budget.

Prediction: with BOS present in training, BOS-prefixed and EOS-prefixed scores agree
within 0.005 for every arm, and the R2 minus R1 difference under BOS lies between the
two earlier values. We report the result whatever it is; the earlier runs stay in the paper.
