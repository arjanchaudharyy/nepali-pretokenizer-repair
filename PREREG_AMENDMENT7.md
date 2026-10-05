# Amendment 7: Qwen R0 at learning rate 3e-4 (written before running)

Written 2026-10-06 after the amendment-6 results. At learning rate 3e-4, Qwen R1 improved from 0.5609
to 0.5092 and matched-steps R2 from 0.5491 to 0.5035, within 0.003 of R0 at 1e-4 (0.5005). R0 was not
run at 3e-4, so the comparison with no extension is not yet like for like.

Run: Qwen R0, learning rate 3e-4, seed 0, otherwise as the main runs (1,095 steps). Remaining budget
under the Rs 10,000 cap of amendment 6.

Predictions:
- P7a: R0 at 3e-4 is no better than R0 at 1e-4 by more than 0.005 (R0 adds no new rows to train).
- P7b: matched-steps R2 at 3e-4 is non-inferior to R0 at 3e-4 (interval entirely below +0.01).

Analysis as in amendment 6.
