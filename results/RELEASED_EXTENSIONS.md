# Released vocabulary extensions vs. the pre-token floor

Regenerate with `.venv/bin/python released_extensions.py` (writes `results/released_extensions.json`, about 1 minute on CPU).
Only `tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json` are downloaded; no weights, no `trust_remote_code`.

Data: FLORES-200 **devtest** of the target language, one sentence per line, NFC, `add_special_tokens=False`, summed over 1,012 sentences.
Base tokenizers come from the same mirrors as `tok_registry.py` (`NousResearch/Meta-Llama-3-8B`, `unsloth/Meta-Llama-3.1-8B`, `unsloth/Llama-3.2-1B`, `Qwen/Qwen2.5-7B`, `Qwen/Qwen3-14B-Base`, `LiquidAI/LFM2-8B-A1B`).

Definitions:
- **Floor** is the pre-token count under the extension's own pre-tokenizer: its normalizer (if any), then `pre_tokenizer.pre_tokenize_str`. Added tokens are not extracted, so this is the regex floor.
- **Ext / floor** is extended tokens divided by the floor. It is at least 1 for any merges-only extension that keeps the regex.
- **Gap closed** is (base - ext) / (base - floor).
- **Repaired floor** is the pre-token count after applying `repaired()` from `box/retrofit_tok.py` to the extension's own regex (PAT_B-style: `\p{L}+` becomes `[\p{L}\p{M}]+`).
- **Headroom** is floor divided by repaired floor.
- **Premium** is target tokens divided by `eng_Latn` devtest tokens, using the same tokenizer.

## A. Merges-based extensions that keep the letters-only regex (the claim)

| Repo | Base | Lang | Vocab base → ext | Base tok | Ext tok | Floor | Ext / floor | Gap closed | Repaired floor | Headroom | Premium base → ext |
|---|---|---|---|---|---|---|---|---|---|---|---|
| atsuki-yamaguchi/Llama-3-8B-te-30K-5000-mean | Llama-3-8B | tel_Telu | 128,256 → 133,256 | 225,274 | **67,303** | 66,222 | **1.016** | 99.3% | 20,964 | 3.16x | 8.29 → 2.33 |
| atsuki-yamaguchi/Llama-3-8B-si-30K-5000-mean | Llama-3-8B | sin_Sinh | 128,256 → 133,256 | 234,504 | 62,117 | 60,836 | 1.021 | 99.3% | 25,132 | 2.42x | 8.63 → 2.22 |
| atsuki-yamaguchi/Llama-3-8B-my-30K-5000-mean | Llama-3-8B | mya_Mymr | 128,256 → 133,256 | 315,795 | 82,873 | 81,791 | 1.013 | 99.5% | 13,935 | 5.87x | 11.62 → 2.95 |
| atsuki-yamaguchi/Llama-3.1-8B-ta-madlad-mean-tuned | Llama-3.1-8B | tam_Taml | 128,256 → 138,256 | 207,768 | 73,639 | 72,365 | 1.018 | 99.1% | 20,210 | 3.58x | 7.65 → 2.65 |
| atsuki-yamaguchi/Llama-3.1-8B-bn-madlad-mean-tuned | Llama-3.1-8B | ben_Beng | 128,256 → 138,256 | 158,715 | 63,073 | 61,861 | 1.020 | 98.7% | 22,570 | 2.74x | 5.84 → 2.26 |
| atsuki-yamaguchi/Llama-3.1-8B-gu-madlad-mean-tuned | Llama-3.1-8B | guj_Gujr | 128,256 → 138,256 | 208,278 | 62,202 | 60,639 | 1.026 | 98.9% | 24,104 | 2.52x | 7.67 → 2.22 |
| atsuki-yamaguchi/Qwen2.5-7B-si-madlad-mean-tuned | Qwen2.5-7B | sin_Sinh | 151,665 → 161,665 | 189,956 | 62,454 | 61,280 | 1.019 | 99.1% | 25,576 | 2.40x | 6.88 → 2.10 |
| atsuki-yamaguchi/Qwen2.5-7B-my-madlad-mean-tuned | Qwen2.5-7B | mya_Mymr | 151,665 → 161,665 | 246,611 | 85,201 | 82,240 | 1.036 | 98.2% | 14,384 | 5.72x | 8.93 → 3.00 |
| atsuki-yamaguchi/Qwen3-14B-Base-bn-madlad-mean-tuned | Qwen3-14B-Base | ben_Beng | 151,665 → 161,669 | 138,814 | 63,465 | 62,298 | 1.019 | 98.5% | 23,007 | 2.71x | 5.03 → 2.15 |
| atsuki-yamaguchi/Qwen3-14B-Base-te-madlad-mean-tuned | Qwen3-14B-Base | tel_Telu | 151,665 → 161,669 | 193,195 | 68,440 | 66,672 | 1.027 | 98.6% | 21,414 | 3.11x | 6.99 → 2.30 |
| LiquidAI/LFM2.5-8B-A1B (Smith et al. 2026) | LFM2-8B-A1B | hin_Deva | 64,400 → 125,017 | 163,668 | 68,983 | 64,563 | 1.068 | 95.5% | 28,788 | 2.24x | 5.91 → 2.50 |
| LiquidAI/LFM2.5-8B-A1B | LFM2-8B-A1B | ben_Beng | 64,400 → 125,017 | 259,987 | 76,903 | 61,861 | 1.243 | 92.4% | 22,570 | 2.74x | 9.39 → 2.79 |
| LiquidAI/LFM2.5-8B-A1B | LFM2-8B-A1B | npi_Deva | 64,400 → 125,017 | 159,816 | 69,102 | 59,578 | 1.160 | 90.5% | 21,530 | 2.77x | 5.77 → 2.51 |
| LiquidAI/LFM2.5-8B-A1B | LFM2-8B-A1B | tha_Thai | 64,400 → 125,017 | 243,469 | 57,084 | 31,380 | 1.819 | 87.9% | 5,629 | 5.57x | 8.79 → 2.07 |

Summary:
- **Yamaguchi et al.** 10 extensions over 7 mark scripts and 3 base families (Llama-3, Llama-3.1, Qwen2.5/Qwen3) land at **1.013 to 1.036x** of their floor (median 1.02). Each closes 98.2 to 99.5% of the base-to-floor gap. These extensions add only 5K or 10K entries.
- **LFM2.5** is the 1T-token, 61K-entry industry expansion. It is near the floor for Hindi (1.07x) and Nepali (1.16x), and has more slack for Bengali (1.24x) and Thai (1.82x). It is a multilingual vocabulary, so per-language budgets are smaller.
- **The regex repair** lowers the floor by **2.2 to 5.9x** for every mark script. Myanmar and Thai gain the most.
- **The Telugu figure in ceiling.tex (67,303 vs 66,222 floor, base 225,274) is reproduced exactly by this script.**

## B. Negative control (no combining marks)

| Repo | Lang | Base tok | Ext tok | Floor | Ext / floor | Repaired floor | Headroom |
|---|---|---|---|---|---|---|---|
| atsuki-yamaguchi/Llama-3.1-8B-am-madlad-mean-tuned | amh_Ethi | 206,128 | 31,570 | 20,076 | 1.573 | 20,076 | 1.00x |

Ethiopic is a syllabary of `\p{Lo}` letters with no `\p{M}`. The repair changes nothing (1.00x), and the same recipe that reaches about 1.02x on mark scripts stays at 1.57x here. The floor binds because of the marks, not because of the recipe.

## C. Extensions that do not obey the floor

| Repo | Base | Lang | Mechanism | Ext tok | Own-regex floor | Ext / floor | Notes |
|---|---|---|---|---|---|---|---|
| hishab/titulm-llama-3.2-1b-v2.0 (TituLLMs) | Llama-3.2-1B | ben_Beng | merges + **changed regex** | 31,711 | 22,349 | 1.419 | The floor under the base Llama-3 regex would be 61,861, so it sits at 0.51x of that. The `repaired()` base regex gives 22,570, close to TituLLM's own 22,349. |
| polyglots/SinLlama_v01 | Llama-3-8B | sin_Sinh | **11,080 added_tokens** (no new merges) | 32,722 | 60,836 | **0.538** | Added tokens match before the regex, so they bypass the floor. Still above the repaired floor (25,132). English is unchanged (27,171). |
| Anurag-Tiwari/hindi_updated-llama-tokenizer (grey literature) | Llama-3-8B | hin_Deva | **44,387 added_tokens** | 39,269 | 64,563 | 0.608 | Bypasses the floor, but English explodes from 27,171 to 101,304 tokens (premium 0.39). Added tokens hijack English substrings, so this is a broken release. Reported only as an illustration. |

TituLLM's regex is `...|\s*\w+|...`. Oniguruma's `\w` admits Mn/Mc, so the regex is mark-aware. It also rewrites the rest of the pattern: there is no `\p{N}{1,3}` digit chunking, there are lookaround digit/letter splits, and whitespace handling differs. Its English count changes slightly (27,171 to 27,137), consistent with REVIEW_1 / RETROFIT_NOVELTY T1.

## Notes and caveats

- **Inaccessible:**
  - `MBZUAI/Llama-3-Nanda-10B-Chat` is gated (401 without an HF token, and none is configured here). It is listed in `inaccessible` in the JSON and is picked up automatically if the script is run with a token that has access.
  - `MBZUAI/Nanda-87B-Chat`: not found.
  - `shubhamdawande/Extending-Llama-3-Tokenizer-Hindi`: not found.
  - Purason et al. checkpoints: no HF repos found under that name.
  - Nag et al. 2024: no HF repo found.
- **Excluded after inspection:**
  - These keep the base Llama-3 tokenizer unchanged, so they are not extensions: Cognitive-Lab Gaja-Hindi, almanach/Llama-3-8B-mono-Nepali, Hemg/nepaligpt-llama3-8b, jayasuryajsk/Llama-3-telugu, subhrokomol/Bengali-Llama-3.1-8B-Tokenizer.
  - These are not extensions of a letters-only base: Aananda-giri/LLAMA3-Nepali (a new 50K BPE with no regex), Suchinthana/Extended-LLAMA-Si-Tokenizer (no regex), pasindubg/extended-sinhala-tokenizer (GPT-2 with added tokens).
  - NaolBM/Qwen3-VL-4B geez (21.8K added tokens) is Ethiopic, so the floor is not binding by marks.
- **Yamaguchi extensions are not English-invariant:**
  - Llama-3 Telugu raises English from 27,171 to 28,925 tokens (+6.5%); Qwen3 Telugu from 27,621 to 29,778 (+7.8%).
  - Their merge lists contain every base merge but in a new order, so new merges interleave with old ones. The LFM2.5 merge list is likewise re-ordered and its ids renumbered (63,893 of 64,400 base strings kept).
  - This matters for the paper's "English-lossless" contrast.
- **Floor validity:**
  - For merges-only extensions with an unchanged regex, `ext_tokens >= pretokens` holds in every row.
  - The bound does not apply to added tokens (section C) or to a changed regex. For TituLLM the measured floor is its own regex's.
- **Base Llama-3 Hindi is already near its floor:** 68,527 base tokens vs a 64,563 floor (1.06x). Further Hindi vocabulary under that regex can gain at most about 6%.
- **LFM2.5 variants:** the `LFM2.5-8B-A1B-Base` tokenizer ships a slightly different letters-only regex variant (`'(?i:[sdmt]|ll|ve|re)`, `\s*[\r\n]`, `\s`). The Instruct release measured here keeps LFM2's regex verbatim. `repaired()` applies to both.
- **Which Yamaguchi checkpoint:**
  - For Llama-3-8B, the 30K-5000-mean checkpoint was used. Its tokenizer.json is byte-identical to the -align variant.
  - Tokenizers differ across sample sizes (for example 30K-100 is not 30K-5000) and between `-mean-tuned` and `-mean-cv` (different file hashes). Only the checkpoints listed above were measured.
  - Other variants can be added to `REPOS` in `released_extensions.py`.
