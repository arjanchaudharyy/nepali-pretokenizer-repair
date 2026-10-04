# Literature review: "The Nepali Token Tax"

Compiled 2026-10-04. Every bibliographic fact below was checked against the arXiv API (export.arxiv.org), the ACL Anthology `.bib` endpoint, proceedings.neurips.cc, proceedings.mlr.press, nature.com, unicode.org, or the primary source (GitHub source code, HF `tokenizer.json`). Anything not checked is marked **UNVERIFIED**. BibTeX: `lit/refs.bib` (48 entries).

---

## 0. Bottom line

**Your core mechanism claim has already been published, and the paper that publishes it is about Nepali.** Regmi, Pudasaini & Pun, *"Vowel Signs Are Not Letters: A Pre-tokenization Ceiling on Multilingual Tokenizer Fertility"* (arXiv:2608.26449, 26 Aug 2026, cs.CL) covers, in order:

- the `\p{L}+` vs `[\p{L}\p{M}]+` mechanism;
- FLORES-200;
- 26 languages, 17 of them abugidas;
- a controlled BPE ablation where only the regex differs;
- a per-language "shatter ratio" that tracks mark density;
- downstream 268M-parameter models;
- a census of tokenizers on Hugging Face.

The "4.78 -> 1.58 tokens/word" Arkios number you asked about comes from that paper's regex-only ablation.

You cannot claim the mechanism, the regex-only ablation, or the cross-abugida scaling as novel. You can still claim:

- a wider and newer audit of production tokenizers (about 26, including Llama-4, Qwen3, and probably Qwen3.5);
- a cost and premium framing centred on Nepali against parallel English;
- a decomposition of the production premium into regex and vocabulary parts across real tokenizers;
- the Amharic/Ethiopic control, which nobody has done;
- grapheme-cluster (UAX #29, GB9c) analysis, which they do not do;
- independent replication.

The paper has to be repositioned as a replication-and-extension of Regmi et al.

**Fact-check of your own setup: DeepSeek-V3 is NOT a `\p{L}+` tokenizer.** Its HF `tokenizer.json` (fetched 2026-10-04) uses this word branch:
```
[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
```
Regmi et al. Table 2 also classifies DeepSeek-V3 as mark-aware. Even so, DeepSeek-V3 scores 4.29 Nepali tokens/word in their table, which is worse than Llama-3 (3.76, letters-only). That is a useful data point for you: a mark-aware regex is not enough without vocabulary allocation. Your `tok_registry.py` classifies automatically from the regex, so it should already label DeepSeek-V3 as `mark-regex`. Make sure the prose in the paper matches.

Verified regexes (from primary sources):

| Tokenizer | Word branch | Source |
|---|---|---|
| GPT-2 | ` ?\p{L}+` | `openai/gpt-2/src/encoder.py` L53 |
| cl100k | `[^\r\n\p{L}\p{N}]?+\p{L}++` | `tiktoken_ext/openai_public.py` |
| o200k | `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+...` (two case branches, both include `\p{M}`) | `tiktoken_ext/openai_public.py` |
| Llama-3 | `[^\r\n\p{L}\p{N}]?\p{L}+` | `meta-llama/llama-models/models/llama3/tokenizer.py` L55 |
| Llama-4 | **literally the o200k pattern** (`O200K_PATTERN`) | `meta-llama/llama-models/models/llama4/tokenizer.py` L122 |
| Qwen2.5 / Qwen3 | `[^\r\n\p{L}\p{N}]?\p{L}+` | HF `Qwen/Qwen2.5-7B`, `Qwen/Qwen3-8B` tokenizer.json |
| DeepSeek-V3 | `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+` (mark-aware) | HF `deepseek-ai/DeepSeek-V3` tokenizer.json |

---

## 1. Novelty threat check

### THREAT 1 (critical): Regmi, Pudasaini & Pun 2026, "Vowel Signs Are Not Letters" (arXiv:2608.26449)

- **Metadata (arXiv API):** v1 26 Aug 2026. cs.CL, cs.LG. 14 pp. Code at `github.com/sajalregmi/arkios-tokenizer` and `huggingface.co/sajalregmi4/arkios-tokenizer`. Not peer reviewed as of today.
- **What it claims:**
  - **Mechanism.** The HF ByteLevel pre-tokenizer inherits GPT-2's `\p{L}+`. Matras (Mc/Mn), virama, and anusvara are `\p{M}`. So the regex splits every abugida word at every vowel sign, and BPE cannot merge across pre-token boundaries.
  - **Bound.** For every string s, `|T(s)| >= |P(s)|`. They call this a "training-free lower bound on fertility" and frame it as a "pre-tokenization ceiling".
  - **Shatter ratio.** This is the number of pre-tokens under `\p{L}+` divided by the number under `[\p{L}\p{M}]+`, measured on FLORES-200 devtest for 26 languages. All 17 abugidas are affected, ranging from 1.47x (Tibetan) to 9.02x (Thai). Burmese is 7.58, Tamil 5.66, Nepali 4.09, Hindi 2.95. Latin, Cyrillic, Hangul and Han are exactly 1.00. Vocalised Arabic is 6.50 and pointed Hebrew 6.14.
  - **Controlled ablation.** Two BPE tokenizers with identical settings: 65,536 vocab, 2,000 MB corpus, 50% target-language bytes, the same bytes verified by sha256, and the same NFC normaliser. Only the word class differs. Nepali goes from **4.78 to 1.58 tokens/word (3.03x)**. English fertility moves from 1.276 to 1.303 (+2.1%). They repeat this for Hindi, Bengali, Tamil and Malayalam, and English costs 1.6 to 2.4% across the five.
  - **Production table (Nepali tokens/word).** Each entry is "tokenizer score (regex class)":
    - GPT-2 10.97 (letters-only)
    - cl100k 6.98 (letters-only)
    - Qwen2.5 6.57 (letters-only)
    - Llama-3 3.76 (letters-only)
    - DeepSeek-V3 4.29 (mark-aware)
    - o200k 2.32 (mark-aware)
    - Gemma-2 3.13
    - Mistral NeMo/Tekken 3.17
    - IndicBERTv2 1.58
    - BLOOM 1.72
    - NLLB-200 1.92
    - mT5 2.64
    - Sarvam-1 2.66
  - **Ratio against English.** Llama-3 Nepali/English is 3.76/1.24 (3.03x) and o200k is 2.32/1.23 (1.89x).
  - **Downstream.** Three 268M models that differ only in the tokenizer. The fixed tokenizer gives 4.43% lower held-out Nepali bits per byte at equal compute (0.4080 -> 0.3899).
  - **Ecosystem census.** 3,479 HF repos were examined. Of 1,345 classified text-generation repos, 63.3% carry a letters-only pre-tokenizer, and these account for 72.5% of 30-day downloads.
  - **Prior art.** They state explicitly that the repair itself is prior art, because o200k already uses it.
- **What it does NOT do** (checked against the HTML full text):
  - no Amharic/Ethiopic or other precomposed-syllable control;
  - no Llama-4 and no Qwen3 measured (Qwen3-0.6B appears only in passing; Qwen3.5-9B is mentioned once in an appendix);
  - no UAX #29 grapheme clusters and no GB9c / Indic_Conjunct_Break;
  - production audit covers 13 tokenizers, not about 26;
  - single seed and a single scale;
  - the premium is reported in tokens/word, not as a FLORES sentence-level token ratio or parity per tokenizer.
- **Overlap with you:** the mechanism, the regex-only controlled ablation, the cross-language scaling with mark density, FLORES-200, Nepali as the focal language, and the 4.78/1.58 numbers. This is near-total overlap with your "attribution" and "controlled experiment" contributions.
- **What remains for you:**
  1. A larger, current production audit (about 26 tokenizers including Llama-4, Qwen3/3.5, GPT-5-era o200k, Gemma-3, Kimi, GLM).
  2. A premium metric defined as sentence-level token ratio against parallel English (Petrov-style parity), with bootstrap CIs.
  3. Decomposition of each production tokenizer's premium into a regex-induced floor and a vocabulary-allocation residual. Their DeepSeek-V3 datapoint shows the residual is large.
  4. **The Amharic/Ethiopic negative control.** This is the cleanest genuinely new piece: an abugida whose syllables are precomposed code points, so it has a high premium from vocabulary but no `\p{M}` shatter.
  5. An akshara / extended-grapheme-cluster analysis, including virama conjuncts under GB9c, which a `[\p{L}\p{M}]+` fix still handles differently from a UAX #29 cluster.
  6. Independent replication of their 3.03x number with your own training data and vocab sizes, including several vocab sizes and seeds.

  You must cite them prominently and frame the paper as building on them.

### THREAT 2 (high, same authors): Regmi, Pudasaini & Pun 2026, "Arkios" (arXiv:2608.30092)

- v1 30 Aug 2026, v2 1 Sep 2026. cs.CL, cs.AI. The comment says "Companion paper (tokenizer): arXiv:2608.26449".
- Arkios-1B is a 1.04B-parameter English-Nepali model trained on 150B tokens with the Devanagari-aware byte-level BPE tokenizer (1.69 Nepali tokens/word).
- The HF model card `sajalregmi4/arkios-tokenizer` is the source of the "4.78 -> 1.58" / "2.5x" figures you asked about. The scholarly record for those numbers is arXiv:2608.26449, not a blog.
- **Overlap:** it fixes Nepali tokenization by adding `\p{M}`. It does not audit production tokenizers.

### THREAT 3 (moderate, prior art for the mechanism, 2024): tiktoken GitHub issue #292

- "Combining marks and indic vowel marks within words are being split breaking all indic languages and most languages except English and CJKs", opened 5 May 2024 by `ajaykg` and now closed.
- It proposes exactly `[^\r\n\p{L}\p{N}]?+[\p{L}\p{M}]+`, about two weeks before GPT-4o's release. This is the earliest public statement of the mechanism we found. Cite it as grey-literature priority.

### THREAT 4 (moderate, grey literature): Sander Land, "Small pre-tokenization bugs with a big multilingual price" (Substack, 1 Sep 2026)

- Identifies the `\p{L}`/`\p{M}` bug.
- Says o200k fixed it and that GPT-5.6 still uses o200k.
- Lists as still affected: Llama 3, Qwen 3, GLM-4/5.
- Lists as fixed or never affected: Llama 4, Qwen 3.5+, Kimi K3, DeepSeek V3/V4, recent Mistral.
- Notes that that model's tokenizer uses the derived Alphabetic property, which keeps some Thai vowel signs and drops viramas and combining accents. That is a third behaviour you could test if you have that model token counts.
- Quantitative claim: Qwen 3 uses 4.4x as many tokens per character for Hindi as for English, falling to 2x in Qwen 3.5 after the regex fix.
- **Use:** this is a natural experiment, Qwen3 to Qwen3.5. If Qwen3.5 is in your 26, it is a strong within-family contrast.

### THREAT 5 (moderate): Velayuthan & Sarveswaran, COLING 2025, "Egalitarian Language Representation in Language Models: It All Begins with Tokenizers"

- arXiv:2409.11501 (17 Sep 2024). COLING 2025, pp. 5987-5996.
- **What it does:**
  - shows GPT-2, GPT-4 and Llama 3 pre-tokenizers "unnecessarily break the text" for Tamil, Sinhala and Hindi;
  - uses Compression Ratio and Petrov's Tokenization Parity on FLORES+;
  - trains BPE, Unigram and WordPiece (5k vocab, 150k Samanantar Tamil sentences) with GPT-2 pre-tokenization against whitespace pre-tokenization. GPT-2 pre-tokenization caps the compression ratio at 1.36, and they conclude "compression ratio is primarily determined by the pre-tokenization methodology";
  - proposes Grapheme Pair Encoding, which is BPE over grapheme clusters.
- **What it does not do:**
  - It does not name `\p{M}` or combining marks as the cause. A text search of the PDF finds no "combining" or `\p{` string.
  - Its ablation compares GPT-2 regex against whitespace, not a minimal `\p{L}` to `[\p{L}\p{M}]` change.
  - Tamil only for the training experiment.
  - No Nepali.
- **Overlap:** the general claim that "pre-tokenization dominates for Indic scripts" and the grapheme-aware proposal.
- **Novel for you:** the precise Unicode-category mechanism (but see Threat 1), Nepali, and production-scale measurement.

### THREAT 6 (moderate): Rana et al., "IndicSuperTokenizer" / "MUTANT" (arXiv:2511.03237)

- v1 5 Nov 2025 was titled *IndicSuperTokenizer: An Optimized Tokenizer for Indic Multilingual LLMs*. v2 22 Mar 2026 was retitled *MUTANT: A Recipe for Multilingual Tokenizer Design*.
- Its Table 1 ablation swaps the GPT-2 regex for the Llama-4 regex. Fertility goes from 3.47 to 1.36 for Hindi, 7.08 to 2.24 for Malayalam, and 6.53 to 2.07 for Tamil, which they describe as a "38-40%" improvement in token-to-word ratio.
- They do not attribute this to `\p{M}`, and their regex swap changes more than the mark class.
- **Overlap:** an empirical regex ablation for Indic scripts. **Novel for you:** the minimal single-class ablation and Nepali.

### Other relevant Indic/abugida tokenization work (low to moderate threat)

- **Kapila & Bagavathy 2026**, "Type-Driven Tokenization for Brahmic Scripts" (arXiv:2609.22125). Treats Brahmic orthography as a partial semigroup, uses an Agda-verified `fixToken`, and ships a SentencePiece patch plus a Rust pre-tokenizer that enforces orthographic boundaries. This is grapheme/akshara-aware pre-tokenization, so it is prior art for your proposed fix, if you propose one.
- **Darshana 2026**, "Separate Before You Compress: The WWHO Tokenization Architecture" (arXiv:2603.25309). Uses DFA/regex syllabification for Sinhala and Devanagari before BPE (SGPE), with a "Linguistic Zero-Breakage" guarantee. Akshara-aware pre-tokenization prior art.
- **Krishnan K & Santhiappan 2026**, "A Grapheme-Aware Indic Tokenizer for Tamil" (arXiv:2609.06690). Uses a reversible grapheme-cluster mapping before WordPiece.
- **Brahma et al. 2025**, "MorphTok" (arXiv:2504.10335; TokShop @ ICML 2025). Constrained BPE stops dependent vowels from standing alone as tokens and reports a 1.68% fertility reduction for Hindi and Marathi. This is related but works at the merge level, not the regex.
- **Shrestha & Pradhan 2026**, "Preserving Morphemes: Morphology-Guided Pre-Tokenization for Nepali" (arXiv:2609.33395, 27 Sep 2026). Builds on Regmi et al.'s fix ("the GPT-2 letter class severs every vowel sign from its akshara") and adds an FST morpheme pre-tokenizer for Nepali, giving about 1.1% BPB. This confirms the Nepali community already treats the `\p{M}` fix as established.
- **Shravan 2026**, "BrahmicTokenizer-131K: An Indic-Capable Drop-In Replacement for o200k_base" (arXiv:2605.29379). Vocabulary retrofitting; the abstract does not discuss the regex.
- **Tamang & Bora 2024**, "Evaluating Tokenizer Performance of LLMs Across Official Indian Languages" (arXiv:2411.12240). NSL for 12 LLMs over 22 languages; notes that GPT-4o improves on GPT-4. No mechanism.
- **Thottingal 2026**, blog "The Broken Token: Tokenization for Malayalam Language Models" (27 Feb 2026). Observes vowel signs isolated as dotted-circle tokens and reports GPT-4/Llama-3 at about 15.8 tokens/word for Malayalam. Does not name the regex.
- **Schmidt, Reddy, Tanner & Pinter 2025**, "Boundless BPE" (COLM 2025, arXiv:2504.00178). Its pre-tokenization pattern keeps combining marks with the base letter. This is incidental prior use of a mark-aware class.

### Nepali-specific premium measurements (moderate threat to the "Nepali premium" framing)

- **Kadariya 2026**, "How Well Does AI Understand Nepali? A Multi-Task Benchmark..." (Zenodo/ResearchGate; GitHub `RameshKadariya/nepali-llm-benchmark`; reportedly July 2026). Reports a Nepali/English token ratio on 150 parallel passages:
  - cl100k 4.79x (14,712 English tokens against 70,476 Nepali tokens)
  - Qwen2.5 4.46x
  - Llama 3.1 2.60x
  - GPT-4o 1.62x
  - NLLB-200 1.19x

  It concludes "Tokenizer design, not the Devanagari script, drives the cost". It names no mechanism. The venue, date and DOI are **UNVERIFIED**, because ResearchGate returned 403. **This pre-empts "first measurement of the Nepali token premium".**
- **Shrestha et al. 2025**, "Towards Nepali-language LLMs: Efficient GPT training with a Nepali BPE tokenizer" (arXiv:2512.14585). A 16k Nepali BPE. No regex analysis.
- **Pudasaini et al. 2025**, "NepaliGPT" (arXiv:2506.16399). No tokenizer-premium analysis.
- **Timilsina, Gautam & Bhattarai 2022**, "NepBERTa" (AACL-IJCNLP 2022 short).
- **Rimal & Rimal 2026**, a benchmark of Llama-3.1-8B, Mistral-7B and Qwen3-8B on Romanized Nepali (arXiv:2604.14171). Relevant only as a contrast (Latin-script Nepali avoids the mark issue).

### "Token tax" title collision (naming risk)

Several papers already use the phrase:
- Lundin et al., "The Token Tax" (AfricaNLP 2026)
- Srivastava, "The Tokenizer Tax" (arXiv:2607.24276)
- Mandarapu & Kunkunuru, "...Multilingual Tokenization Tax" (arXiv:2609.00378)
- Somide, "The African Language Tax" (arXiv:2606.24460)
- Serval, "The Invisible Language Tax" (arXiv:2609.39001)
- Ovcharov, "The Tokenizer Tax Across 25 European Languages" (arXiv:2605.24718)

"The Nepali Token Tax" is still distinguishable, but cite Lundin et al. at minimum.

- **Srivastava 2026** (FLORES-200, 10 Indian languages, cl100k/o200k/GPT-2/Qwen2.5/mBERT/XLM-R). Reports Hindi at 4.08x and Malayalam at 13.04x under cl100k. Attributes the tax to the "unmerged single-byte token rate" (r = 0.89), not the regex. It reports that o200k cuts the mean Indic tax from 8.0x to 2.1x without identifying why. **Your mechanism explains their finding.** Cite it as a paper that observed the effect without diagnosing it.
- **Mandarapu & Kunkunuru 2026** (FLORES-200 devtest; English, Spanish, German, Hindi, Bengali, Telugu, Tamil, Kannada; GPT-2/cl100k/o200k plus grapheme clusters). Decomposes the tax into removable and irreducible parts and reports Hindi at 4.76x with removable fraction rho = 0.59. They do not attribute any of it to the regex. Their "removable" component plausibly contains your regex effect, so you can connect the two.

---

## 2. Verified bibliography (one line each)

Key = BibTeX key in `refs.bib`.

### Foundational tokenization
- `sennrich-etal-2016-neural`: Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units*. ACL 2016, pp. 1715-1725, arXiv:1508.07909. Introduces BPE for subword NMT.
- `radford2019language`: Radford, Wu, Child, Luan, Amodei, Sutskever. *Language Models are Unsupervised Multitask Learners*. OpenAI tech report, 2019. This is the GPT-2 byte-level BPE. The regex ` ?\p{L}+` is verified in `openai/gpt-2/src/encoder.py`; the report text describes byte-level BPE and preventing merges across character categories. Cite the code for the exact regex.
- `kudo-richardson-2018-sentencepiece`: Kudo & Richardson. *SentencePiece...*. EMNLP 2018 System Demonstrations, pp. 66-71, arXiv:1808.06226. A language-independent tokenizer with no regex pre-tokenization.
- `kudo-2018-subword`: Kudo. *Subword Regularization...*. ACL 2018, pp. 66-75, arXiv:1804.10959. Introduces the unigram LM tokenizer.

### Cross-lingual disparity
- `petrov2023unfairness`: Petrov, La Malfa, Torr, Bibi. NeurIPS 2023, arXiv:2305.15425. Shows parallel FLORES text differs by up to 15x in length; defines tokenization parity. Pages **UNVERIFIED** and omitted.
- `ahia-etal-2023-languages`: Ahia, Kumar, Gonen, Kasai, Mortensen, Smith, Tsvetkov. *Do All Languages Cost the Same?...*. EMNLP 2023, pp. 9904-9923, arXiv:2305.13707. Shows API cost and utility disparities from tokenization across 22 languages.
- `rust-etal-2021-good`: Rust, Pfeiffer, Vulić, Ruder, Gurevych. ACL-IJCNLP 2021, pp. 3118-3135, arXiv:2012.15613. Finds that a dedicated monolingual tokenizer improves multilingual model performance; introduces the fertility and continued-words metrics.
- `velayuthan-sarveswaran-2025-egalitarian`: see Threat 5.
- `ahia2024magnet`: Ahia, Kumar, Gonen, Hofmann, Limisiewicz, Tsvetkov, Smith. *MAGNET*. NeurIPS 2024, arXiv:2407.08818. Script-specific boundary predictors equalise byte-level segmentation.
- `limisiewicz-etal-2024-myte`: Limisiewicz, Blevins, Gonen, Ahia, Zettlemoyer. *MYTE*. ACL 2024, pp. 15059-15076, arXiv:2403.10691. A morpheme-based byte encoding that reduces cross-lingual length disparity.

### Tokenizer quality vs downstream performance
- `zouhar-etal-2023-tokenization`: Zouhar, Meister, Gastaldi, Du, Sachan, Cotterell. ACL 2023, pp. 5184-5207, arXiv:2306.16842. Proposes Rényi efficiency as a predictor of MT BLEU.
- `cognetta-etal-2024-two`: Cognetta, Zouhar, Moon, Okazaki. *Two Counterexamples to Tokenization and the Noiseless Channel*. LREC-COLING 2024, pp. 16897-16906, arXiv:2402.14614. Random-Drop BPE and Duplicate BPE raise Rényi efficiency but hurt BLEU.
- `schmidt-etal-2024-tokenization`: Schmidt, Reddy, Zhang, Alameddine, Uzan, Pinter, Tanner. *Tokenization Is More Than Compression*. EMNLP 2024, pp. 678-702, arXiv:2402.18376. Fewer tokens does not by itself mean better downstream performance; pre-tokenization matters.
- `goldman-etal-2024-unpacking`: Goldman, Caciularu, Eyal, Cao, Szpektor, Tsarfaty. **Findings of ACL 2024** (not EMNLP, although the arXiv comment says "EMNLP 2024, Findings"; the Anthology ID 2024.findings-acl.134 is authoritative), pp. 2274-2286, arXiv:2403.06265. Compression correlates with downstream performance, more strongly for generation tasks and small models.
- `ali-etal-2024-tokenizer`: Ali et al. (21 authors). *Tokenizer Choice For LLM Training: Negligible or Crucial?*. Findings of NAACL 2024, pp. 3907-3924, arXiv:2310.08754. Tokenizer choice matters for multilingual downstream performance; fertility and parity are not reliably predictive.
- `dagan2024getting`: Dagan, Synnaeve, Rozière. ICML 2024, PMLR 235:9784-9805, arXiv:2402.01035. **Directly relevant:** shows that the pre-tokenization regex is a first-order design choice affecting compression and downstream code performance.
- `liu2025superbpe`: Liu, Hayase, Hofmann, Oh, Smith, Choi. *SuperBPE: Space Travel for Language Models*. COLM 2025, arXiv:2503.13423. Lifting the whitespace pre-tokenization constraint in a second stage gives about 33% fewer tokens and better downstream results. Pre-token boundaries constrain efficiency, the same logic as the "BPE cannot merge back" argument.
- `schmidt2025boundless`: BoundlessBPE. COLM 2025, arXiv:2504.00178. Merges across pretokens into superwords.
- `lundin-etal-2026-token`: Lundin, Zhang, Karim, Louzan, Wei, Adelani, Carroll. *The Token Tax: Systematic Bias in Multilingual Tokenization*. AfricaNLP 2026, pp. 103-112, arXiv:2509.05486. Fertility predicts AfriMMLU accuracy across 10 LLMs and 16 African languages; doubling tokens quadruples training cost.
- `meister2026tokeval`: Meister. *TokEval: A Tokenizer Evaluation Suite*. COLM 2026, arXiv:2608.18062. Information-theoretic intrinsic metrics predict LM ability (Spearman rho up to 0.80); structure-sensitive metrics predict task accuracy. A 2026 follow-up to the Rényi-efficiency line.

### Pricing and premium audits (2026)
These entries cover the token-count side only; the pricing pages themselves were not needed and were not fetched.
- `roy2026premium`: Roy, Roy, Patel. IJCAI 2026 workshop, arXiv:2608.09046. Uses a 120-item Python tutoring corpus. Bengali costs 1.56x GPT-4o tokens and up to 4.5x on Qwen2.5/Mistral. **No Nepali** and no mechanism.
- `somide2026african`: Somide. arXiv:2606.24460. Covers African languages, including **Amharic (Ge'ez/Ethiopic)**, and reports a 7 to 9x penalty for Ethiopic and N'Ko on frontier tokenizers, plus a 7.4x latency multiplier for Amharic on GPT-5. **Relevant to your Amharic control:** Ethiopic has a high premium without any `\p{M}` shatter. Use it to show that your control separates the regex effect from the vocabulary effect.
- `serval2026french`, `ovcharov2026european`, `srivastava2026tokenizertax`, `mandarapu2026ledger`: see above.

### Data
- `goyal-etal-2022-flores`: Goyal, Gao, Chaudhary, Chen, Wenzek, Ju, Krishnan, Ranzato, Guzmán, Fan. *The Flores-101 Evaluation Benchmark...*. TACL 10:522-538 (2022), arXiv:2106.03193.
- `nllb2022`: NLLB Team et al. *No Language Left Behind: Scaling Human-Centered Machine Translation*. arXiv:2207.04672 (2022). Introduces FLORES-200.
- `nllb2024nature`: NLLB Team. *Scaling neural machine translation to 200 languages*. Nature 630:841-846 (2024), doi:10.1038/s41586-024-07335-x. The peer-reviewed FLORES-200/NLLB citation.
- Note: "FLORES+" (OLDI) is the community-maintained continuation used by Velayuthan & Sarveswaran. State which release and split you used (`npi_Deva`, `eng_Latn`, `amh_Ethi`, devtest/dev).

### Nepali
- `timilsina-etal-2022-nepberta`: Timilsina, Gautam, Bhattarai. *NepBERTa: Nepali Language Model Trained in a Large Corpus*. AACL-IJCNLP 2022 (short), pp. 273-284.
- `shrestha2025nepalibpe`, `pudasaini2025nepaligpt`, `shrestha2026papaya`, `regmi2026arkios`, `regmi2026vowel`, `rimal2026romanized`, `kadariya2026nepali` (partially **UNVERIFIED**): see above.

### Unicode
- `uax29_151`: UAX #29 *Unicode Text Segmentation*, Revision 43 (Unicode 15.1.0, 2023-08-16, ed. Josh Hadley). **Verified:** GB9c is absent from Revision 41 (Unicode 15.0.0, 2022-08-26) and present in Revision 43. Rule text in Revision 43:
  ```
  \p{InCB=Consonant} [\p{InCB=Extend}\p{InCB=Linker}]* \p{InCB=Linker} [\p{InCB=Extend}\p{InCB=Linker}]* × \p{InCB=Consonant}
  ```
  It is defined via the Indic_Conjunct_Break property. GB9 (× Extend|ZWJ) and GB9a (× SpacingMark) keep matras attached to their base.
- `uax29_180`: Revision 49 (Unicode 18.0.0, 2026-09-01). **GB9c was revised again** to `\p{InCB=Linker} \p{InCB=Extend}* × \p{InCB=Consonant}` (UTC 187-C47). If you report grapheme counts, pin the Unicode version of your segmenter (for example, the Python `regex` module's `\X`, or ICU), because the counts for conjuncts can differ between 15.0, 15.1+ and 18.0.

### o200k regex change and its effect on Indic text
- **No peer-reviewed paper documents OpenAI's change.** The evidence is:
  - the tiktoken source (verified above);
  - tiktoken issue #292 (May 2024, proposing the change);
  - tiktoken issue #298 (18 May 2024, by AmitMY, asking about o200k regex flavour and what changed from cl100k; closed with no visible answer);
  - Regmi et al. 2026, who name o200k as mark-aware;
  - Land 2026 (blog);
  - Srivastava 2026, who measures the drop from 8.0x to 2.1x without attributing it.
- OpenAI's GPT-4o launch materials report fewer tokens for Indic languages, but they were **not verified here** and no regex is mentioned. A Microsoft Tech Community blog reports Hindi falling from 2,090 tokens (GPT-4) to 655 (GPT-4o); this is **UNVERIFIED** as to methodology.

---

## 3. Recommended repositioning

1. **Title and abstract.** Drop the "we identify the mechanism" framing. Use something like: "Regmi et al. (2026) showed the `\p{L}` ceiling. We ask how much of the *production* Nepali premium it explains across 26 deployed tokenizers, and we validate it with a precomposed-abugida control."
2. **Lead contributions:**
   - (a) a current audit across tokenizers, including Llama-3 against Llama-4 and Qwen3 against Qwen3.5 as within-family natural experiments;
   - (b) a premium decomposition into a regex floor plus a vocabulary residual, using DeepSeek-V3 (mark-aware but 4.29 tokens/word) as the motivating counterexample;
   - (c) the Amharic/Ethiopic control;
   - (d) a GB9c/akshara analysis;
   - (e) replication of the 3.03x ablation across several vocab sizes and seeds.
3. **Correct the DeepSeek-V3 regex classification** in your text.
4. **Must-cite list:** Regmi et al. 2026 (both papers), tiktoken #292, Velayuthan & Sarveswaran 2025, Rana et al. (IndicSuperTokenizer/MUTANT), Petrov 2023, Ahia 2023, Srivastava 2026, Mandarapu & Kunkunuru 2026, Kadariya 2026, Lundin 2026, Dagan 2024, Land 2026 (as a blog).

## 4. Unverified / caveats
- Kadariya 2026: venue, date and DOI are UNVERIFIED (ResearchGate 403). The numbers come from the project's GitHub README.
- Page numbers for the NeurIPS papers (Petrov 2023, MAGNET 2024) are UNVERIFIED and omitted.
- Some content summaries were produced by a summarising fetcher over arXiv HTML. The key numbers for Threat 1 were cross-checked across three separate fetches, but re-read the PDF of arXiv:2608.26449 before quoting it. One early summary wrongly listed DeepSeek-V3 as letters-only; the verbatim-quote fetch and the HF `tokenizer.json` both show it is mark-aware.
- The GPT-4o launch-page claims about Indic token reductions were not fetched.
