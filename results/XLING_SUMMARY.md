# Cross-lingual K sweep: does the mark-aware regex repair generalize? (compression only)

Same protocol as `results/ksweep.json` (Nepali), CPU only, no model training.
Premium = sum(language tokens) / sum(English tokens) over the 1012 paired FLORES-200 devtest sentences (NFC),
with paired bootstrap 95% CI (B=2000, seed 0). Floor = pre-token count of the language devtest under that arm's regex,
divided by base English tokens (no vocabulary can go below it). Mark density = share of NFC code points in `\p{M}`.
R0 = base tokenizer; R1 = continued BPE under the original regex; R2 = mark-aware regex (`repaired()` / `PAT_B`) + continued BPE.
R1 floor is also the floor of R0 (same regex). Raw rows: `results/xling_ksweep.json`; per-cell files `results/xl/`.

## Main table (K = 32,000)

| Language | Base | Mark density | R0 premium | R1 floor | R1@32k | R2@32k | R2 floor | English identical | Lossless |
|---|---|---|---|---|---|---|---|---|---|
| Hindi | Llama-3.2-1B | 0.3083 | 2.5221 [2.4936, 2.5509] | 2.3762 | 2.3790 [2.3517, 2.4071] | 1.1676 [1.1572, 1.1782] | 1.0595 | True (6/6) | True |
| Hindi | Qwen3-1.7B-Base | 0.3083 | 4.4160 [4.3609, 4.4712] | 2.3538 | 2.3566 [2.3291, 2.3850] | 1.1647 [1.1542, 1.1753] | 1.0585 | True (6/6) | True |
| Bengali | Llama-3.2-1B | 0.3154 | 5.8413 [5.7755, 5.9069] | 2.2767 | 2.2851 [2.2607, 2.3105] | 1.0076 [0.9982, 1.0172] | 0.8307 | True (6/6) | True |
| Bengali | Qwen3-1.7B-Base | 0.3154 | 5.0257 [4.9650, 5.0861] | 2.2555 | 2.2643 [2.2388, 2.2895] | 1.0048 [0.9955, 1.0141] | 0.8330 | True (6/6) | True |
| Tamil | Llama-3.2-1B | 0.3530 | 7.6467 [7.5550, 7.7344] | 2.6633 | 2.6681 [2.6379, 2.6963] | 1.0169 [1.0066, 1.0273] | 0.7438 | True (6/6) | True |
| Tamil | Qwen3-1.7B-Base | 0.3530 | 6.1123 [6.0343, 6.1866] | 2.6357 | 2.6403 [2.6096, 2.6699] | 1.0169 [1.0066, 1.0273] | 0.7474 | True (6/6) | True |
| Telugu | Llama-3.2-1B | 0.3582 | 8.2910 [8.1850, 8.3970] | 2.4372 | 2.4525 [2.4241, 2.4807] | 1.0169 [1.0052, 1.0285] | 0.7716 | True (6/6) | True |
| Telugu | Qwen3-1.7B-Base | 0.3582 | 6.9945 [6.8988, 7.0882] | 2.4138 | 2.4288 [2.3996, 2.4577] | 1.0069 [0.9954, 1.0178] | 0.7753 | True (6/6) | True |
| Sinhala | Llama-3.2-1B | 0.2813 | 8.6307 [8.5311, 8.7327] | 2.2390 | 2.3000 [2.2748, 2.3253] | 1.1963 [1.1832, 1.2100] | 0.9250 | True (6/6) | True |
| Sinhala | Qwen3-1.7B-Base | 0.2813 | 6.8772 [6.7929, 6.9623] | 2.2186 | 2.3282 [2.3015, 2.3557] | 1.2423 [1.2272, 1.2580] | 0.9260 | True (6/6) | True |
| Thai | Llama-3.2-1B | 0.2055 | 2.1942 [2.1689, 2.2212] | 1.1549 | 1.4921 [1.4754, 1.5097] | 0.9543 [0.9430, 0.9663] | 0.2072 | True (6/6) | True |
| Thai | Qwen3-1.7B-Base | 0.2055 | 2.5508 [2.5183, 2.5850] | 1.1530 | 1.4818 [1.4654, 1.4991] | 0.8796 [0.8694, 0.8902] | 0.2207 | True (6/6) | True |
| Burmese | Llama-3.2-1B | 0.5008 | 11.6225 [11.4738, 11.7774] | 3.0102 | 3.0245 [2.9865, 3.0644] | 1.2067 [1.1947, 1.2195] | 0.5129 | True (6/6) | True |
| Burmese | Qwen3-1.7B-Base | 0.5008 | 8.9284 [8.8030, 9.0624] | 2.9774 | 2.9913 [2.9516, 3.0327] | 1.1063 [1.0951, 1.1177] | 0.5208 | True (6/6) | True |
| Amharic (control) | Llama-3.2-1B | 0.0000 | 7.5863 [7.4966, 7.6747] | 0.7389 | 0.9763 [0.9667, 0.9857] | 0.9763 [0.9667, 0.9857] | 0.7389 | True (6/6) | True |
| Amharic (control) | Qwen3-1.7B-Base | 0.0000 | 4.0454 [3.9966, 4.0950] | 0.7431 | 0.9760 [0.9667, 0.9851] | 0.9760 [0.9667, 0.9851] | 0.7431 | True (6/6) | True |
| Korean (control) | Llama-3.2-1B | 0.0000 | 1.4883 [1.4739, 1.5030] | 0.6936 | 0.9755 [0.9651, 0.9855] | 0.9755 [0.9651, 0.9855] | 0.6936 | True (6/6) | True |
| Korean (control) | Qwen3-1.7B-Base | 0.0000 | 1.6299 [1.6125, 1.6469] | 0.6990 | 0.9757 [0.9655, 0.9856] | 0.9757 [0.9655, 0.9856] | 0.6990 | True (6/6) | True |

"English identical" = English token ids identical to the base on all 1012 English devtest sentences, for all 6 builds (R1, R2 x 3 K).
"Lossless" = decode(encode(s)) == s on all 1012 language + 1012 English devtest sentences, for all 6 builds.
Nepali reference (results/ksweep.json, 1 GB training text): R0 2.5832 / 4.4154, R1 floor 2.1927 / 2.1733, R1@32k 2.2040 / 2.1869, R2@32k 1.0144 / 1.0144, R2 floor 0.7924 / 0.7958 (Llama / Qwen3).

## Full K sweep (premium; R1@32k column gives the number of target-script merges actually available)

| Language | Base | R1@4k | R1@16k | R1@32k (merges) | R2@4k | R2@16k | R2@32k |
|---|---|---|---|---|---|---|---|
| Hindi | Llama-3.2-1B | 2.3965 | 2.3813 | 2.3790 (31,521) | 1.4717 | 1.2405 | 1.1676 |
| Hindi | Qwen3-1.7B-Base | 2.3797 | 2.3588 | 2.3566 (32,000) | 1.4991 | 1.2400 | 1.1647 |
| Bengali | Llama-3.2-1B | 2.3231 | 2.2895 | 2.2851 (32,000) | 1.4129 | 1.0947 | 1.0076 |
| Bengali | Qwen3-1.7B-Base | 2.2987 | 2.2682 | 2.2643 (32,000) | 1.3958 | 1.0896 | 1.0048 |
| Tamil | Llama-3.2-1B | 2.6815 | 2.6687 | 2.6681 (22,398) | 1.4294 | 1.1206 | 1.0169 |
| Tamil | Qwen3-1.7B-Base | 2.6533 | 2.6410 | 2.6403 (22,325) | 1.4216 | 1.1193 | 1.0169 |
| Telugu | Llama-3.2-1B | 2.4777 | 2.4543 | 2.4525 (25,862) | 1.4648 | 1.1215 | 1.0169 |
| Telugu | Qwen3-1.7B-Base | 2.4534 | 2.4305 | 2.4288 (25,794) | 1.4460 | 1.1090 | 1.0069 |
| Sinhala | Llama-3.2-1B | 2.3497 | 2.3033 | 2.3000 (25,220) | 1.5852 | 1.2809 | 1.1963 |
| Sinhala | Qwen3-1.7B-Base | 2.3769 | 2.3315 | 2.3282 (25,180) | 1.6233 | 1.3255 | 1.2423 |
| Thai | Llama-3.2-1B | 1.7873 | 1.5767 | 1.4921 (32,000) | 1.3281 | 1.0594 | 0.9543 |
| Thai | Qwen3-1.7B-Base | 1.7855 | 1.5658 | 1.4818 (32,000) | 1.1629 | 0.9664 | 0.8796 |
| Burmese | Llama-3.2-1B | 3.0598 | 3.0285 | 3.0245 (31,815) | 1.7635 | 1.3432 | 1.2067 |
| Burmese | Qwen3-1.7B-Base | 3.0227 | 2.9945 | 2.9913 (30,728) | 1.5808 | 1.2194 | 1.1063 |
| Amharic (control) | Llama-3.2-1B | 1.3902 | 1.0761 | 0.9763 (32,000) | 1.3902 | 1.0761 | 0.9763 |
| Amharic (control) | Qwen3-1.7B-Base | 1.3759 | 1.0733 | 0.9760 (32,000) | 1.3759 | 1.0733 | 0.9760 |
| Korean (control) | Llama-3.2-1B | 1.2319 | 1.0547 | 0.9755 (32,000) | 1.2319 | 1.0547 | 0.9755 |
| Korean (control) | Qwen3-1.7B-Base | 1.2376 | 1.0547 | 0.9757 (32,000) | 1.2376 | 1.0547 | 0.9757 |


## Training text (FineWeb-2 TEST split only, all surviving text used)

| Language | Docs | Dropped (10-gram) | Dropped (char-50) | Training text (bytes) | Parquet (bytes) |
|---|---|---|---|---|---|
| Hindi | 55,242 | 1 | 0 | 325,437,594 | 88,669,788 |
| Bengali | 59,078 | 0 | 0 | 361,287,599 | 94,475,656 |
| Tamil | 41,100 | 0 | 0 | 297,057,154 | 71,465,017 |
| Telugu | 18,933 | 0 | 0 | 150,550,619 | 38,920,910 |
| Sinhala | 11,428 | 0 | 0 | 75,354,448 | 20,304,606 |
| Thai | 24,037 | 0 | 0 | 200,016,532 | 51,285,472 |
| Burmese | 15,491 | 0 | 0 | 131,193,903 | 31,327,996 |
| Amharic (control) | 3,973 | 0 | 0 | 25,471,018 | 8,583,199 |
| Korean (control) | 30,074 | 0 | 0 | 113,423,681 | 54,028,075 |

## Findings

1. **The R1 floor reproduces in every mark-heavy abugida.** Hindi, Bengali, Tamil, Telugu, Sinhala and Burmese all saturate within ~0.01 to 0.11 of their R1 floor (2.22 to 3.01) by K=4k to 16k, and in Hindi(Llama), Tamil, Telugu, Sinhala and Burmese continued BPE under R1 *runs out of mergeable pairs* (fewer than 32,000 merges exist at min_frequency 2). R2 reaches 1.0048 to 1.2423 at K=32k on the same text. Burmese is the extreme: R0 11.6225 (Llama), R1 floor 3.0102, R2@32k 1.2067.
2. **Controls behave as predicted.** Amharic (precomposed Ethiopic) and Korean (precomposed Hangul) have mark density 0.0000, R1 and R2 produce *identical* premiums at every K (same pre-tokens), and R1 is not floored (R1@32k 0.976, floor ~0.74). The floor effect is caused by combining marks, not by abugidas or non-Latin scripts per se.
3. **Thai is a partial case.** Thai has marks (density 0.2055) but most vowels are letters (Lo), so its R1 floor is only 1.15 and R1 is not saturated at 32k (1.4921 / 1.4818). R2 still wins (0.9543 / 0.8796). Its R2 floor (0.21) is tiny because Thai writes no spaces between words, so whole phrases become one pre-token; the same holds for Burmese (R2 floor ~0.51).

## Anomalies and caveats

- **Hindi shares Devanagari with Nepali.** Its target-script filter is the same block, so Hindi-trained merges also apply to Nepali text (and vice versa); this cross-effect was not measured. Hindi R2@32k (1.1676) is worse than Nepali's (1.0144) because its R2 floor is higher (1.0595 vs 0.7924): Hindi FLORES sentences have more short function-word pre-tokens per English token. Not a script effect.
- **Unequal, much smaller training text than Nepali** (25 MB Amharic to 361 MB Bengali, vs 1 GB Nepali). R2@32k for Sinhala (75 MB) and Amharic (25 MB) is likely data-limited; cross-language R2@32k comparisons are confounded by corpus size. All test splits are above 20 MB of text except Amharic's parquet (8.6 MB parquet, 25.5 MB text).
- **No-space scripts (Thai, Burmese):** sentence-level token ratios are fine, but word-level measures are meaningless. Whitespace 10-gram decontamination can only see 69/1012 Thai and 562/1012 Burmese FLORES sentences, so a supplementary char-50-gram check (drops any doc sharing a >=74-char span with FLORES dev/devtest) was run for all languages; it found 0. The 10-gram pass dropped 1 doc (Hindi). Self-test: the 10-gram detector flags 1002/1002 Hindi FLORES sentences with >=10 words.
- **Builder change needed for poorly covered scripts.** The original filter keeps a merge only if its decoded output contains a complete target-script character. In Llama-3, Telugu/Sinhala/Burmese/Amharic characters are mostly 2 to 3 byte tokens, and merges like `" " + lead byte` were discarded, killing every later merge that depended on them (first run: Sinhala R2 got only 13,530 of 32,000 merges, 30,463 dropped as dependents). `box/retrofit_tok_xl.py` additionally admits merges whose non-ASCII bytes are fragments of target-script characters (`FRAG=1`, default; `FRAG=0` restores the original rule). These tokens contain a non-ASCII byte and cannot match ASCII text; English identity holds in all 108 builds. Fragment merges kept per 32k run: 0 to 140 (most in Korean).
- **K-prefix builds.** One training run per (language, base, arm) with K=32,000; the K=4k/16k tokenizers are prefixes of its filtered merge list (greedy BPE order does not depend on the target vocab size).
- The R2 regex change itself alters pre-tokenization of every mark-bearing script, so an R2 tokenizer is not identical to the base on other Indic languages (only English identity is guaranteed and tested here).

Files: `box/retrofit_tok_xl.py`, `box/prep_xl.py`, `box/decontam_char_xl.py`, `box/xl_sweep.sh`, `box/xl_eval.py`; tokenizers in `tok/xl/<lang>/<model>/<arm>_K<k>/`; logs in `logs/xl/`.
