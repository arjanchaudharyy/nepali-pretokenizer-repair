# Retrofit direction: novelty threats, baselines, evaluation, licensing

Compiled 2026-10-04. This file extends `LIT_REVIEW.md` and does not repeat it.

**How claims were checked.** Every bibliographic claim was checked against at least one primary source: the arXiv API (export.arxiv.org), ACL Anthology `.bib`, CrossRef DOI, mlanthology (TMLR), or the paper PDF read locally with `pdftotext`. Claims about what a paper "does not mention" come from a full-text grep for `pre-?tokeniz|regex|\p{|combining|diacritic|vowel sign`. Claims about released tokenizers come from downloading the model's `tokenizer.json` from Hugging Face and running it with HF `tokenizers`. Anything not checked is marked **UNVERIFIED**.

All new BibTeX (37 entries) is appended to `refs.bib` under a dated comment.

---

## 0. Verdict

**Repairing a deployed letters-only model (`\p{L}+`) for Nepali with a regex fix, compared head-to-head against the standard vocabulary-extension recipe, is still unclaimed.** I found no paper that does all three of the following:

1. takes a deployed byte-level BPE model with the letters-only regex;
2. changes its pre-tokenizer to admit `\p{M}`;
3. continues training and attributes the gain to the regex, against an R1-style control.

The closest prior work is in three places:

- **TituLLMs** (Findings ACL 2025) silently shipped this repair for Bangla. This is the single biggest threat; see T1.
- **ZeTT** (NeurIPS 2024) used a mark-aware regex for target tokenizers in an appendix.
- **Several 2024–2026 vocabulary-expansion papers on exactly our model families** (Llama-3/3.1, Qwen2.5/3, LFM2) kept the letters-only regex and were therefore capped without noticing. These papers are evidence for our hypothesis, not threats to it.

**Measured here: the R1 ceiling on FLORES-200 devtest `npi_Deva`** (18,701 whitespace words, HF `tokenizers`):

| Regex | Original pre-tokens/word | Repaired `[\p{L}\p{M}]+` | Current tokens/word |
|---|---|---|---|
| Llama-3 regex | 3.186 | 1.151 | n/a |
| Qwen3 (`Qwen/Qwen3-1.7B`) | 3.210 | 1.175 | 6.521 |

Tokens per word cannot fall below pre-tokens per word. So R1 is floored at about 3.2 tokens/word whatever K is, while R2's floor is about 1.15.

Regmi et al. report Llama-3 at 3.76 tokens/word (their word count). That is already within about 18% of the 3.19 floor, so **R1 on Llama-3 has very little headroom.** This is a strong, cheap headline.

English pre-tokens were identical under both regexes for **1012/1012** FLORES `eng_Latn` devtest sentences, with both the Llama-3 and Qwen3 patterns.

---

## 1. Novelty threats (retrofit-specific)

The four columns are:

- **(a)** targets a byte-level model with the letters-only regex;
- **(b)** notices or addresses the `\p{M}` ceiling;
- **(c)** shows English tokenization is preserved;
- **(d)** measures Nepali or another abugida.

### T1 (HIGH): TituLLMs. Nahin et al., Findings of ACL 2025, pp. 24922-24940.

Bib key `nahin-etal-2025-titullms`; arXiv:2502.11187.

| (a) | (b) | (c) | (d) |
|---|---|---|---|
| **Yes**: Llama-3.2-1B/3B | **Implicitly yes, never stated** | **No, it changes English** | **Bangla** |

**What they did.** They extended the Llama-3.2 tokenizer by about 42K tiktoken-trained Bangla tokens (vocabulary 128K to about 170K, `hishab/titulm-llama-3.2-{1b,3b}-v2.0`) and continued pretraining on about 37B tokens. Bangla tokens/word went from 7.84 to 1.90 (their Table 5). The paper says only that they "modified the original Tiktoken codebase to ... better accommodate the morphological complexities of Bangla" (Sec. 3).

**What their released tokenizer actually does.** The released `tokenizer.json` **replaces the Llama-3 pre-tokenizer regex** with a different pattern whose word branch is `\s*\w+`. Under the Oniguruma engine `\w` includes marks. Running it, `আমি বাংলায় কথা বলি` pre-tokenizes into 4 whole words, where Llama-3 produces 11 mark-split fragments. So they **did** repair the mark split in a deployed Llama-3.2 model and then ran CPT.

**But:**

1. **They never identify the regex or `\p{M}` as the cause.** Their 7.84 to 1.90 gain mixes regex and vocabulary, with no regex-only or vocabulary-only arm.
2. **Their regex is not English-lossless.** I verified each of these against the Llama-3 regex:
   - digits are no longer split into groups of three (`2024`, `12345` become single pre-tokens);
   - `snake_case_var` becomes one pre-token;
   - runs of whitespace attach to the following word.

   So English and code token ids change.
3. **Not Devanagari.** Their regex also fails on Sinhala ZWJ conjuncts: `ශ්‍රී` splits at the ZWJ.
4. **No decontrolled comparison** against vocabulary-only extension.

**Implication.** You cannot claim to be "the first to change the pre-tokenizer of a deployed `\p{L}+` model and continue training". You can claim the first:

- **controlled** attribution, with R1 vs R2 at the same K and data;
- **minimal, English-lossless** repair, with a proof plus a 1012/1012 check;
- **Nepali/Devanagari** result, with a quantified R1 ceiling.

Cite TituLLMs explicitly, and frame their tokenizer as an uncredited instance of the fix. Report that their regex changes English and digit pre-tokens.

### T2 (MEDIUM-HIGH, priority on the idea): ZeTT. Minixhofer, Ponti & Vulić, NeurIPS 2024

Bib key `minixhofer2024zett`.

**What it says.** Appendix I ("Assumptions on the Tokenization Function"), verbatim:

> "pretokenization given by a regular expression based on the regular expression used by GPT2 ..., adjusted to not over-segment text in languages using characters in the Unicode Mark category within words (e.g. Hindi and Tamil)."

The code confirms it: `SPLIT_REGEX = r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"` (`bminixhofer/zett`, `zett/utils.py` L29).

| (a) | (b) | (c) | (d) |
|---|---|---|---|
| **No**: Mistral-7B, TinyLlama-1.1B, XLM-R are all SentencePiece, so there is no `\p{L}+` bug to repair | **Yes, as a one-sentence design choice** | n/a | XNLI languages incl. Hindi (encoder); no Nepali |

**Implication.** The mark-aware pre-tokenizer as a target for transferring a pretrained model already exists, and Minixhofer et al. noticed over-segmentation for Hindi and Tamil. They did not apply it to repair a letters-only model and did not measure the effect. Cite as priority, and note that ZeTT-style hypernetwork transfer is a possible R2 alternative to FVT plus CPT.

### T3 (MEDIUM, is our R1 recipe; supports H1): Purason, Chizhov, Yamshchikov & Fishel, "Teaching Old Tokenizers New Words", Findings of EACL 2026, pp. 6492-6516

Bib key `purason-etal-2026-teaching`.

| (a) | (b) | (c) | (d) |
|---|---|---|---|
| **Yes**: Llama-3, Llama-3.2-1B/3B, Qwen-2.5, Mistral-NeMo | **No** (grep: only generic "apply the pretokenizer") | **Yes**: Table 6, English exact-match ≥97.7% after extension | Tokenizer-only compression over 70 FineWeb-2 languages **including `npi_Deva`**, `hin_Deva`, Tamil, Thai, Burmese, Khmer, Sinhala (App. Tables 13-18); CPT only for Estonian |

**What it does.** It introduces continued BPE training (append merges learned on top of the existing merge list), plus leaf-based pruning, with FVT initialisation. Their Nepali numbers are R1-style: the regex is not touched, so they are capped.

**Implication.** This is the method your R1/R2 merge-learning uses, so cite it as the source. Their per-language tables (App. Table 17/18) give an independent R1 compression figure for Nepali on Llama-3. Extract it and show it sits above the 3.19 floor. **UNVERIFIED:** I did not extract the exact Nepali cell from the column-mangled PDF text. Read it from the PDF.

### T4 (MEDIUM, industry R1 at scale; supports H1): Smith et al. (Liquid AI), "In-Place Tokenizer Expansion for Pre-trained LLMs", arXiv:2607.15232 (16 Jul 2026)

Bib key `smith2026inplace`.

**What it does.** It expands LFM2-8B-A1B from 65K to 128K by continued BPE merges, with mean-of-source-rows initialisation and 600B embedding-only plus 400B full CPT. It reports Hindi 2.4x and Bengali 3.35x token-count reductions, and Nepali appears in its Global-MMLU table.

**Their App. C states "Byte-level pre-tokenization rules are inherited."** I fetched `LiquidAI/LFM2-8B-A1B` and `LFM2.5-1.2B-Base` `tokenizer.json`: both use the **Llama-3 letters-only regex**. So a 1T-token production retrofit left the pre-token ceiling in place.

| (a) | (b) | (c) | (d) |
|---|---|---|---|
| Yes | **No** (no mention of marks or regex) | "English and code stay at parity by design" | Hindi, Bengali, Thai, Nepali (eval only) |

**Implication.** This is a strong motivating citation: well-resourced industry vocabulary expansion did not touch the regex.

### T5 (MEDIUM, canonical academic R1 on our model families and abugidas; supports H1): Yamaguchi et al.

- **"How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB...?"** Computational Linguistics 52(1):295-330, 2026. Bib key `yamaguchi2026lowres`.
- **"Adapting Chat Language Models Using Only Target Unlabeled Language Data" (ElChat).** TMLR 2025. Bib key `yamaguchi2025elchat`.

**Coverage.** Vocabulary expansion with mean initialisation on **Llama-3-8B, Llama-3.1-8B, Qwen2.5, Qwen3** for Telugu, Sinhala, Burmese, Tamil, Bengali, Gujarati, Thai and Amharic. Their own blog post on byte-level vocabulary expansion (gucci-j.github.io, 2024-06-24) rebuilds the BPE with new merges and keeps the Llama-3 regex.

**Demonstration on their release.** On `atsuki-yamaguchi/Llama-3-8B-te-30K-5000-mean`, a 7-word Telugu sentence gives **13 pre-tokens and 13 tokens**. The expanded vocabulary has hit the ceiling exactly.

| (a) | (b) | (c) | (d) |
|---|---|---|---|
| Yes | **No** | not as a tokenization claim | abugidas yes, Nepali no |

**Implication.** These are the best-known peer-reviewed R1 baselines. Use them as the R1 recipe citation and as "prior VE on abugidas was capped".

### T6 (LOW-MEDIUM): other Indic and abugida adaptations of byte-level models

Each was checked against its released `tokenizer.json` or paper text. None changes the regex deliberately; TituLLMs (T1) is the only exception.

- **Nanda (Llama-3, Hindi).** arXiv:2504.06011, later "Nanda Family", EACL 2026, bib `singh-etal-2026-nanda`.
  - Adds about 20% Hindi tokens: fertility 2.6x down to 1.19, with English preserved.
  - Uses WECHSEL-style similarity initialisation.
  - The paper has no regex discussion. The HF repo is gated, so the released regex is **UNVERIFIED**.
- **SinLlama (Llama-3-8B, Sinhala).** MERCon 2025.
  - Adds 11,080 tokens as **HF `added_tokens`**, not merges. 10,124 of them contain `\p{M}` characters.
  - **Added tokens are matched before the regex runs**, so this accidentally *bypasses* the ceiling for those strings. Example: `ශ්‍රී ලංකාව` becomes whole-word tokens even though the regex still splits.
  - Note this as a third mechanism, an "R1b" arm worth a sentence: added-token bypass is greedy, non-BPE, and does not compose.
- **Nemotron-3 Hindi embedding-init study.** Joshi et al., arXiv:2608.03494.
  - The base Nemotron-3-Nano tokenizer is **o200k-style mark-aware** (verified), so no ceiling applies.
  - It is still useful as the strongest 2026 Hindi init study: uniform subword mean plus Hindi norm calibration for input embeddings, char-length-weighted mean for output embeddings, a 6x step reduction, and "Init loss is unreliable".
- **Typhoon 1.5 (Llama-3, Thai), OpenThaiGPT 1.5 / Sailor2 (Qwen2.5).** All ship the unmodified letters-only regex and did not extend the vocabulary. Thai is the worst-shattered language (Regmi: 9.02x).
- **Tibetan CPT of Qwen2.5.** Yang et al., arXiv:2507.09205. The abstract has no regex discussion. Whether they extended the vocabulary is **UNVERIFIED**; one grep hit mentions "pre-tokenization strategy" in a data-processing context.
- **Llama-2 / SentencePiece adaptations** (OpenHathi, Airavata, Tamil-Llama, Kannada Llama, SambaLingo, EEVE, Mundra et al., Tejaswi et al.) are **out of scope for (a)**. SentencePiece has no `\p{L}` regex, so these adaptations never faced the bug. Cite them for the recipe only.
  - Tejaswi et al. (Findings EMNLP 2024) is the standard "vocabulary extension + CPT design choices" reference, using Hindi and Tamil.
- **Nag et al., arXiv:2412.10244 (Llama-3, Indic VE).** Adds tokens learned with SentencePieceBPE and uses Gee et al. mean initialisation. No regex discussion.
- **Duwal et al., arXiv:2412.13860 (Llama-3-8B, Nepali).** QLoRA CPT for Nepali with **no vocabulary change**. Notes Nepali's high fertility under Llama-3. This is the only prior Llama-3-for-Nepali adaptation paper found, and it is an R0-like baseline.
- **Grey literature.** `shubhamdawande/Extending-Llama-3-Tokenizer-Hindi` (Jul 2024) keeps the letters-only regex.

### T7 (LOW): BrahmicTokenizer-131K. Shravan, arXiv:2605.29379

- **It is a tokenizer-only retrofit of o200k_base**, which is already mark-aware. "The pre-tokenizer, decoder, and inherited merge rules are unchanged from o200k_base."
- **No model training:** "we do not perform model-side training to consume the new vocabulary."
- No `\p{M}`, regex or Nepali discussion. It is relevant only for the "English preserved bit-identically" framing, which it also claims.

| (a) | (b) | (c) | (d) |
|---|---|---|---|
| No (source is mark-aware) | No | Yes | 11 Indian languages, not Nepali |

### T8 (LOW): IndicSuperTokenizer/MUTANT (arXiv:2511.03237)

Already covered in `LIT_REVIEW.md` (Threat 6). It trains from scratch rather than retrofitting.

### T9 (LOW): Dagan, Synnaeve & Rozière, ICML 2024 (`dagan2024getting`, already in refs)

**This is the precedent for changing a pretrained model's pre-tokenizer during continued training.** It fine-tunes Code Llama with tokenizers that differ in the pre-tokenization regex and finds that more than 50B tokens of training recover the loss. It is not multilingual, and it is not about marks.

Cite it for "regex changes on a pretrained model are survivable with CPT".

### Other tokenizer-transfer work (cite, no overlap on the regex)

- WECHSEL, FOCUS and OFA are init methods (Sec. 2).
- **ALM** (NeurIPS 2025) does cross-tokenizer distillation.
- **Retrofitting with Dynamic Tokenization** (Feher et al., 2024).
- **OMP transplantation** (Goddard & Fernandes Neto, 2025).
- **VocADT** (ICLR 2025).
- **HyperOFA** (ACL SRW 2025).
- **Tokenizer swapping** (Dobler & de Melo, WANT@ICML 2024).

None mentions `\p{M}`; ZeTT is the only one that does (T2).

### Design notes these papers surfaced (for the methods section)

**1. "English-lossless" needs the precise statement.**

- **The theorem.** For any string containing no `\p{M}` code point, `[\p{L}\p{M}]+` and `\p{L}+` give identical pre-tokens. New merges then cannot fire on ASCII pre-tokens, so English ids are identical.
- **Where it fails.** It is false for NFD Latin. I verified that `café`, `piña` and `x̲` all change.
- **Where it holds.** Emoji with VS16 (`❤️`) are unchanged.
- **What to report.** The fraction of English *documents* in your CPT/eval corpora that contain any `\p{M}`, and whether you NFC-normalise.

**2. The Llama-3/Qwen regex is not GPT-2's.**

The optional prefix `[^\r\n\p{L}\p{N}]?` lets one mark attach to the following letter run. So the deployed pre-tokens are "mark + letters" (for example `ाल`), and old merges spanning a mark followed by a consonant exist in the vocabulary.

Under R2 (and R3), old merges therefore fire on the now-contiguous words, producing **old-token sequences the model never saw** for Nepali. This is the R3 confound. Report the fraction of FLORES Nepali tokens under R3 that are bigrams the base model never saw, or at least acknowledge it.

**3. Use merges, not `added_tokens`.**

R1/R2 must add merges, not HF `added_tokens`. Otherwise R1 silently bypasses the regex, as SinLlama does. Your PREREG already uses continued BPE, which is correct; say so explicitly.

---

## 2. Baselines and methods to compare against or cite

### Embedding initialisation

FVT/mean is the default and is fine as the primary choice. Cite the alternatives, and run one stronger init as a robustness arm if budget allows.

| Method | Key | Use |
|---|---|---|
| FVT (mean of constituent old-token embeddings) | `gee-etal-2022-fast` (EMNLP 2022 Industry, pp. 409-416) | **Primary init.** Also used by Purason, Liquid, Nag, Yamaguchi. |
| Mean-init analysis / convex hull, CW2V | `mundra-etal-2024-empirical` (CoNLL 2024, pp. 84-104) | Theory that convex-hull init is good. Shows simple mean is competitive. |
| WECHSEL | `minixhofer-etal-2022-wechsel` (NAACL 2022, pp. 3992-4006) | Bilingual-dictionary/fastText init; used by Nanda. |
| FOCUS | `dobler-de-melo-2023-focus` (EMNLP 2023, pp. 13440-13454) | Overlap-anchored sparsemax init. **Best "strong init" arm**, with code available. |
| OFA / HyperOFA | `liu-etal-2024-ofa`, `ozeren2025hyperofa` | Factorised / hypernetwork init. |
| ZeTT | `minixhofer2024zett` | Hypernetwork; alternative to FVT plus CPT for R2. |
| Init-strategy study (Hindi, 2026) | `joshi2026beyondinit` | Norm calibration; decoupled input/output init; "init loss unreliable, use a 50-step probe". **Cite; consider their input-norm calibration.** |

### Vocabulary extension and CPT recipes (R1 lineage)

| Recipe | Key |
|---|---|
| Continued BPE extension. **Your R1/R2 merge learner.** | `purason-etal-2026-teaching` |
| In-place expansion at production scale | `smith2026inplace` |
| VE + mean init on Llama-3/Qwen for abugidas, low-data regime | `yamaguchi2026lowres`, `yamaguchi2025elchat` |
| VE + CPT design choices (Hindi, Tamil) | `tejaswi-etal-2024-exploring` |
| EEVE (staged parameter freezing for VE) | `kim2024eeve` |
| SambaLingo (VE + CPT, 9 languages) | `csaki2024sambalingo` |
| Tokenizer swapping on a budget | `dobler2024tokenizerswap` |
| Changing the pre-tokenizer of a pretrained model | `dagan2024getting` (existing) |
| Indic examples | `choudhury2025nanda` / `singh-etal-2026-nanda`, `nahin-etal-2025-titullms`, `aravinda2025sinllama`, `gala2024airavata`, `balachandran2023tamilllama`, `nag2024efficientcpt` |
| Nepali CPT (no VE) | `duwal2024nepalidapt` |

**Recommended baseline set for the paper.** R0, R1 (continued BPE under the original regex, which is the Purason/Yamaguchi recipe), R2, and R3 if affordable.

Add a single **FOCUS-init R2** arm if budget allows, to show the result is not an artefact of mean init.

Optionally add an "R1b: added-tokens bypass" tokenizer-only row in the compression table. It is cheap and pre-empts a reviewer who asks "why not just `add_tokens`?".

---

## 3. Evaluation resources for Nepali (verified)

Nepali config availability was checked via the HF datasets-server `/splits` API.

| Resource | Nepali config | Citation key | Notes |
|---|---|---|---|
| **Belebele** | `npi_Deva` (and `npi_Latn`) | `bandarkar-etal-2024-belebele` (ACL 2024, pp. 749-775) | 900 MCQ, parallel with `eng_Latn`. |
| **SIB-200** | `npi_Deva` | `adelani-etal-2024-sib` (EACL 2024) | Topic classification, 7 classes. |
| **FLORES-200** | `npi_Deva` | `nllb2022`, `nllb2024nature`, `goyal-etal-2022-flores` (existing) | Already local at `flores200_dataset/`. `openlanguagedata/flores_plus` is gated on HF. |
| **Global-MMLU** | `ne` | `singh-etal-2025-global` (ACL 2025) | Apache-2.0, not gated. Liquid AI used it for Nepali. |
| **MMLU-ProX** | `ne` | `xuan-etal-2025-mmlu` (EMNLP 2025) | 29 languages; probably too hard for 1-2B models (near chance). |
| **IndicGenBench** | Nepali (`ne`) listed | `singh-etal-2024-indicgenbench` (ACL 2024) | Generation (FLORES-IN, CrossSum-IN, XQuAD-IN, XorQA-IN). Nepali per-task coverage **UNVERIFIED**. |
| **NLUE** (supersedes Nep-gLUE) | 12 tasks | `nyachhyon-etal-2025-consolidating` (Findings IJCNLP-AACL 2025) | NLU classification/inference; suited to encoders but usable with prompting. |
| Kadariya 2026 multi-task Nepali benchmark | | `kadariya2026nepali` (existing) | Venue **UNVERIFIED**. |

### How Regmi and Arkios scored Belebele

- **Regmi et al. 2608.26449 has no Belebele.** Their downstream metric is held-out BPB only (FineWeb-2 Nepali, C4 English).
- **Arkios 2608.30092, Sec. 7.2** reports both formats, zero-shot:

  | Format | Nepali | English |
  |---|---|---|
  | Letter-choice (lm-eval-harness, A/B/C/D) | 0.240 | 0.236 |
  | Answer-text scoring ("scores each answer string directly, without a letter mapping") | 0.306 | 0.387 |

  Letter format is at chance for a 1B model.

**Recommendation.** Report answer-text log-likelihood as the primary Belebele metric, with letter format in the appendix, matching Arkios.

**Tokenizer confound.** For R1/R2 vs R0 the tokenization differs, so use **length-normalised log-likelihood per byte** (acc_norm by bytes), not per token. Otherwise the tokenizer change itself shifts the scores. State this explicitly. It is the main way a reviewer could attack the downstream comparison.

---

## 4. Licensing (verified 2026-10-04)

### Qwen3 (Qwen3-0.6B/1.7B/4B, Base and post-trained)

- **Apache-2.0.** The HF API `cardData.license` is `apache-2.0`, the repos are not gated, and `Qwen/Qwen3-1.7B-Base/LICENSE` is the Apache License 2.0 text.
- Derivative fine-tunes and research release are permitted. Keep the LICENSE file and NOTICE, and state your modifications.

### Llama-3.2 (1B/3B)

**Llama 3.2 Community License** (release date 25 Sep 2024; `meta-llama/llama-models/models/llama3_2/LICENSE`). The HF repo is gated with manual approval.

- **Sec. 1.a** grants the right to "use, reproduce, distribute, copy, create derivative works of, and make modifications to the Llama Materials". Research release of a CPT derivative is allowed.
- **Obligations under Sec. 1.b:**
  - ship a copy of the Agreement;
  - display "Built with Llama";
  - **the model name must begin with "Llama"** (for example `Llama-3.2-1B-NepaliRepair`);
  - include the NOTICE line "Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.";
  - comply with the Acceptable Use Policy.
- **Sec. 2** requires a separate licence above 700M MAU, which is irrelevant here.
- **AUP:** the EU restriction applies **only to multimodal Llama 3.2 models**. The 1B/3B text models are unaffected (`USE_POLICY.md`, verified).

### Other target families (not license-checked here)

GLM, Phi-4 and OLMo-2 are **UNVERIFIED** for licence; check each before release. Regex verification for GLM-4.5, Phi-4 and OLMo-2-1124-7B is done: all ship the letters-only `\p{L}+` regex (`tokenizer.json`, verified).

---

## 5. Positioning sentence (suggested)

> Regmi et al. (2026) showed that a `[\p{L}\p{M}]+` pre-tokenizer lowers Nepali BPB when a model is trained from scratch, and ZeTT (Minixhofer et al., 2024) already used such a regex for tokenizer transfer. Yet every published vocabulary-expansion retrofit of Llama-3/Qwen models for abugidas that we inspected kept the letters-only regex, which floors Nepali at 3.2 tokens/word on FLORES whatever vocabulary is added (Yamaguchi et al. 2025, 2026; Purason et al. 2026; Smith et al. 2026). The one exception, TituLLMs (Nahin et al., 2025), changed the regex without attribution and in a way that alters English and digit tokenization. We give the first controlled comparison of regex repair against vocabulary extension for a deployed model, with a repair that provably leaves every mark-free (e.g., all-ASCII) English string's token ids unchanged.
