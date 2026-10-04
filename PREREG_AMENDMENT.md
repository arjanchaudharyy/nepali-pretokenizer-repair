# Pre-registration amendment 2 (before any CPT result was observed)

Written 2026-10-04, after an internal review and before any continued-pretraining
run had produced an evaluation. Earlier CPT launches were stopped by us before
completion because of pipeline bugs (below); none produced an evaluated
checkpoint. The only evaluation run before this amendment is of the untouched
base Llama-3.2-1B with a since-fixed document-prefix bug; it is discarded.

## Known before registration (not predictions)

P1 (compression ordering and the R1 floor) was measured before the first
addendum. We report it as a measurement, not a confirmed prediction.

## Deviations from PREREG_RETROFIT.md, and why

| Registered | Now | Reason |
|---|---|---|
| Qwen3-1.7B-Base | Qwen3-0.6B-Base | Budget (authors' own money) and time. Same tokenizer. |
| 3 GB + 3 GB per arm | 1 GB Nepali + 1 GB English per arm | Budget and time. Identical documents in every arm. |
| LR chosen per model on R0 from {3e-5, 1e-4, 3e-4} by 200-step pilot | Fixed 1e-4 for every run | Budget and time. The pilot was never completed. 1e-4 is a common CPT rate. We will not tune it on results. |
| 2 seeds per arm | 1 seed (seed 0) per arm; a second Llama seed only if budget allows | Budget and time. |
| (none) | R0 and R1 also checkpointed at R2's step count | Equal-compute view requested by review. Their cosine schedules are not annealed at that point; we report this. |
| (none) | Byte-matched BPB: held-out documents split into ~2,000-byte pieces, each scored independently | Token windows give R2 more bytes of context; this removes that confound. |
| Translation cap 256 tokens | 512 tokens | Avoids truncating R0 Nepali output. |

## Pipeline bugs fixed before any reported run

1. Documents were stored one per line with internal newlines, so "documents" were paragraphs. Now one JSON-encoded document per line.
2. New-token embedding init decoded partial-UTF-8 tokens to U+FFFD. Now constituents come from the base merges applied to the token's byte-level string.
3. The init exactness check covered Qwen's padding rows. Now only the original vocabulary.
4. Evaluation prefixed documents with end-of-text; base Llama expects BOS. Now BOS where it exists, else end-of-text, identically for every arm of a model.

## Primary analysis (fixed now)

- Primary metric: Nepali bits per byte on the held-out native Nepali slice, **byte-matched** version (2,000-byte pieces). The sliding-window version is secondary.
- Comparison: R2 vs R1 and R2 vs R0 at equal data (final checkpoints).
- Test: paired bootstrap over held-out pieces (B = 10,000, seed 0) of the per-piece bits-per-byte difference; report the mean difference and its 95% interval.
- Claim rules:
  - "R2 better" if the 95% interval of (R2 minus other) lies entirely below 0;
  - "R2 non-inferior" if it lies entirely below +0.01 bits per byte;
  - otherwise "no detectable difference".
  The 0.01 margin is about 0.4% of base Nepali BPB, smaller than any difference we would call practically meaningful.
- With one seed we cannot separate arm effects from data-order variance; we say so wherever we make a claim.
- English control: the same test on English pieces; "no harm" if the interval of (R2 minus R0) lies below +0.01.
- Secondary, reported regardless of outcome: FLORES BPB, Belebele (answer-text log-likelihood per byte, with a 95% bootstrap interval), chrF++ both directions, generation bytes per second, and the equal-compute checkpoints.
