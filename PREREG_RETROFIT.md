# Pre-registration addendum: repairing deployed models

Written 2026-10-04, after discovering that Regmi, Pudasaini & Pun (arXiv:2608.26449)
had already published the mechanism, the regex-only ablation, the cross-script
analysis and a from-scratch 268M downstream comparison. The original PREREG.md
hypotheses H1–H4 are therefore withdrawn as novel contributions; where we still
report them, it is as replication.

Before this addendum: the continued-BPE retrofit tokenizers were built and
checked on a 30 MB Nepali sample (FineWeb-2 npi_Deva *test* split) only to test the
code. No model has been trained or evaluated with them.

## Question

Most widely downloaded open models (Llama-3.x, Qwen2.5/3, GLM, Phi-4, OLMo-2)
ship a `\p{L}+` pre-tokenizer. Can such a model be repaired for Nepali after
deployment, and is a regex repair needed, or does the standard Indic recipe
(vocabulary extension + continued pretraining) suffice?

## Arms (identical except as stated)

- **R0**: original tokenizer, continued pretraining (CPT). Controls for "more Nepali training".
- **R1**: K new merges by continued BPE under the model's own regex + CPT. The standard recipe.
- **R2**: K new merges by continued BPE under the repaired regex + CPT.
- **Base**: the model with no CPT, as a reference.

For R1 and R2: the same Nepali text, the same K, the same restriction (every new
merge must output Devanagari) and the same embedding initialisation (each new
row is the mean of its base-token constituents' rows, for both input and output
embeddings).

## Fixed protocol

- Models: Llama-3.2-1B (base) and Qwen3-1.7B-Base. If budget allows, one larger
  model (Qwen3-4B-Base or Llama-3.1-8B), one seed, labelled as such.
- K = 32,000 unless the compression sweep shows R2 has saturated earlier. The
  choice is made from compression on FLORES **dev**, never from model quality.
- CPT data: a fixed byte budget per model, 50/50 by bytes Nepali/English
  (FineWeb-2 npi_Deva, FineWeb-Edu), FLORES-decontaminated, the identical
  byte stream for every arm. *Equal data* is the primary comparison; each arm's
  training FLOPs are reported.
- Optimisation: identical for all arms. Sequence length 2048, global batch
  ~0.5M tokens, AdamW, cosine schedule, 1% warmup. The learning rate is chosen
  once on R0 for each model and reused for R1 and R2.
- Seeds: 2 per arm for the two primary models.

## Primary outcome and hypotheses

- **P1 (compression).** On FLORES devtest, Nepali premium: R2 < R1 < R0, with R1
  bounded below by the letter-regex pre-token floor (2.16× Qwen3, 2.19× Llama-3).
- **P2 (main).** Held-out native Nepali BPB after CPT: R2 ≤ R1 + 0.01 and
  R2 ≤ R0 + 0.01 (non-inferiority, 0.01 bits/byte), while R2 uses fewer training
  tokens. Superiority is also tested and reported either way.
- **P3 (no English harm).** English token ids are bit-identical across R0/R1/R2
  by construction (already verified). After CPT, English BPB differs between
  R2 and R0 by < 0.01 bits/byte.
- **P4 (practical).** At equal bytes generated, R2 decodes Nepali faster
  (bytes/second) and fits more Nepali bytes in a fixed context window.

**Secondary (exploratory):** Belebele npi_Deva and eng_Latn zero-shot, scored
on the answer text (not letter choice; see Arkios, arXiv:2608.30092); FLORES
ne→en 5-shot chrF++; the GPU-hours needed to repair.

## What would count against us

- If R2 is worse than R0 on Nepali BPB beyond the margin, the repair costs model
  quality, and compression alone does not justify it.
- If R1 matches R2 on BPB despite worse compression, the ceiling exists at the
  tokenizer level but does not matter for modelling at this budget. We would
  report that as the finding.
