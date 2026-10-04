# Pre-registration: The Nepali Token Tax

Written 2026-10-04, before any tokenizer in Experiments 2–4 was trained and before
any language model was trained. Experiment 1 (the production sweep,
`results/sweep_devtest.json`) had already been run and motivates these hypotheses;
it is observational and is not used to test them.

Authors: Aarjan Chaudhary, Shrey Sharma, Navyata Dhakal.

## The claim under test

GPT-2-style regex pre-tokenizers whose word class is `\p{L}+` exclude Unicode
combining marks (`\p{M}`). In Devanagari every dependent vowel sign, virama,
anusvara, candrabindu and nukta is a combining mark. Such a pre-tokenizer therefore
starts a new pre-token at every vowel sign, so BPE can never learn a merge that
joins a consonant to its own vowel sign. We call the resulting excess token
count, relative to the same tokenizer with marks admitted, the *mark tax*.

## Fixed definitions

- **Premium** of language L under tokenizer T: tokens_T(L) / tokens_T(eng) summed
  over the same FLORES-200 sentences (devtest unless stated). 95% CI by paired
  bootstrap over sentences, B = 2000, seed 0.
- **Arm A (letter)**: the Llama-3 / cl100k pre-tokenization regex, verbatim.
- **Arm B (mark)**: Arm A with every word class `\p{L}` widened to admit `\p{M}`
  (also inside the negated leading-character class). Nothing else differs:
  same corpus, same vocabulary size, same BPE trainer, same byte-level alphabet,
  same normalisation (NFC), same special tokens.
- **Mark density** of a language: share of NFC code points in its FLORES devtest
  text that are `\p{M}`.
- **BPB** (bits per byte): summed token negative log-likelihood in bits / UTF-8
  bytes of the scored text. This is comparable across tokenizers; per-token loss
  is not.

## Hypotheses and the test of each

**H1 (mechanism, compression).** At fixed corpus and vocabulary size, Arm B lowers
the Nepali premium relative to Arm A. Tested at vocab ∈ {16k, 32k, 64k, 128k} and
Nepali share of tokenizer-training bytes ∈ {5%, 50%}. Reported as the ratio
premium_A / premium_B with bootstrap CI. We will call H1 supported if the ratio's
95% CI excludes 1 at every grid point.

**H2 (negative control).** Arm B changes English tokens per byte by less than 1%
at every grid point. English has almost no combining marks; a large English change
would mean the intervention does something other than what we claim.

**H3 (dose–response across scripts).** For each language in a fixed set of 14
(listed in `xling.py`; includes Amharic, an abugida whose vowels are precomposed
letters rather than marks, and Korean, Russian as non-abugida controls), train an
Arm A and an Arm B tokenizer on that language + English at identical settings. The
log reduction log(premium_A / premium_B) increases with mark density. Test:
Spearman rho across languages, one-sided, alpha = 0.05; we also report Pearson r
and the fit. Prediction: Amharic, Korean and Russian show reductions near zero.

**H4 (downstream, the main result).** Decoder-only LMs trained from scratch on a
50/50 (by bytes) Nepali/English mixture, identical except Arm A vs Arm B
tokenizer at 64k vocabulary:
- (a) *equal compute* (same tokens seen): Arm B reaches lower Nepali BPB on
  held-out native Nepali text and on FLORES devtest.
- (b) *equal data* (same bytes seen): Arm B reaches Nepali BPB no worse than Arm A
  (non-inferiority margin 0.01 bits/byte) while using fewer training FLOPs.
- (c) English BPB differs between arms by less than 0.01 bits/byte (control).
Scales: ~125M, ~350M, ~1B parameters; seeds 3 / 2 / 1. Effect sizes reported
with seed-level spread; at 1B we report a single run and say so.

**H5 (compute equivalence).** Fitting Nepali BPB against training FLOPs per arm
(power law over the three scales), Arm A needs a multiplier k > 1 of Arm B's
compute to reach the same Nepali BPB. We report k with an interval from the fit.
This is exploratory if fewer than three scales complete.

**Secondary (exploratory, labelled as such in the paper).** Belebele npi_Deva
zero-shot accuracy (likely near chance below 1B; reported regardless). Inference
cost per byte. Effective context in bytes.

## What would count against us

- H1 ratio CI including 1 at large vocabularies: the mark tax is a small-vocab
  artefact.
- H3 rho near zero or Amharic showing a large reduction: the mechanism is not
  marks.
- H4(a) null: compression gains do not translate into modelling gains at these
  scales. We will report this as the main finding if it happens.

## Decontamination and splits

Every training document (tokenizer and LM) sharing a whitespace 10-gram with
FLORES-200 dev or devtest in any studied language is dropped. Belebele and SIB-200
are built from FLORES passages, so this also covers them. Exact-duplicate
documents are removed. A native Nepali and English held-out slice is carved
before training and never trained on. FLORES dev is used for any tuning;
devtest is reported.
