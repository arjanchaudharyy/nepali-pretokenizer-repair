# Amendment 6: second seeds for every key arm, and a learning-rate check (written before running)

Written 2026-10-06, after the amendment-5 results. Budget cap: Rs 10,000 of GPU time. Runs are queued
in the order below and the queue stops when the cap is reached; every completed run is reported.

## Runs (settings as in the main runs unless stated; seed changes data order only)

1. Qwen R2, matched steps (748), seed 1.
2. Llama R2, matched steps (737), BOS setting, seed 1.
3. Llama R1, BOS setting, seed 1.
4. Llama R0, BOS setting, seed 1.
5. Qwen R0, seed 1.
6. Qwen R1, learning rate 3e-4, seed 0.
7. Qwen R2, matched steps, learning rate 3e-4, seed 0.

## Scoring and analysis

As in amendment 5: byte-matched held-out Nepali BPB after the registered prefix, document-clustered
bootstrap, claim rules in fixed order. Seed-averaged comparisons average the per-piece bits of the two
seeds of each arm before bootstrapping.

## Predictions

- P6a: with seeds averaged, matched-steps R2 is better than R1 on Qwen (interval entirely below 0).
- P6b: with seeds averaged, matched-steps R2 is worse than R1 on Llama (interval entirely above 0).
- P6c: R0 is best on both models with seeds averaged.
- P6d: at learning rate 3e-4, matched-steps R2 - R1 on Qwen has the same sign as at 1e-4.

If a prediction fails, we report it and change the paper's claims accordingly.
