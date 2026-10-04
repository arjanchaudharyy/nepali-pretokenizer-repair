# Pre-registration amendment 3: Experiment B (embedding warm-up, more data)

Written 2026-10-04 after observing Experiment A's Llama result (R2 worse than R0 on
held-out Nepali BPB by +0.066 [+0.065, +0.067], also at equal compute; English
equal) and BEFORE observing Experiment A's R1 result or any Qwen result.
Experiment A is reported in full regardless of what follows.

## Motivation (stated before running)

R1 and R2 add 32,000 new rows initialised as means of existing rows, then train
them for one epoch of 1 GB + 1 GB with the whole model at lr 1e-4. Vocabulary-
extension work commonly trains new embeddings alone first (EEVE, Kim et al.
2024; in-place expansion, Smith et al. 2026). Hypothesis: Experiment A's R2 deficit
is largely under-trained new embeddings, and shrinks with an embedding warm-up
stage and with more data.

## Design (Llama-3.2-1B; identical across arms)

- Stage 1: train only the tied embedding matrix, lr 1e-3, on the first 10% of the
  arm's token stream (same 10% of documents in every arm).
- Stage 2: full continued pretraining from the stage-1 model, lr 1e-4, one epoch over
  the arm's full stream, otherwise as Experiment A.
- B1: 1 GB Nepali + 1 GB English (the Experiment A streams). Arms R0, R1, R2.
- B3: 3 GB + 3 GB (new streams, same cleaning; first 3 GB of each language). Arms R0
  and R2, plus R1 if budget allows.
- Seed 0. Same evaluation and the same primary test as amendment 2.

## Hypotheses and claim rules

- HB1: the R2 minus R0 Nepali BPB difference in B1 is smaller than in A (+0.066).
- HB2: the R2 minus R0 difference in B3 is smaller than in B1.
- Claim rules for each comparison as in amendment 2 (better / non-inferior /
  no detectable difference / worse, 95% paired bootstrap interval, margin 0.01).
- We will report the trend across A, B1, B3 whatever it shows, including if R2
  remains worse. A trend from one seed and three budgets is reported as
  suggestive, not established.
