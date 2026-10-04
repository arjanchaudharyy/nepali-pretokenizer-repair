# Tokenizer extras (review W4, W6, W10, N7)

Regenerate: `.venv/bin/python tokenizer_extras.py` (CPU, about 65 s) -> `results/tokenizer_extras.json`.
Inputs: `tok/box/<model>/R{1,2}_K32000/` (copied from the box), `results/ksweep_meta/` (all 28 K-sweep `meta.json`), FLORES-200 devtest (1,012 sentences), `data/ne_test_sample.txt`.

## 1-2. Compression: regex-only (R2_K0) and added-tokens (R1b) arms

Premium = Nepali tokens / English tokens of the same tokenizer, paired bootstrap 95% CI (B=2000, seed 0, as in `box/ksweep_eval.py`). Every row below is lossless on all 1,012 Nepali and English sentences and gives English ids identical to the base tokenizer.

| Arm | Llama-3.2-1B | Qwen3-1.7B-Base |
|---|---|---|
| R0 (base) | 2.583 [2.554, 2.612] | 4.415 [4.358, 4.475] |
| **R2_K0** (repaired regex, no new merges) | **2.608 [2.579, 2.636]** | **4.415 [4.358, 4.475]** |
| R1_K32000 (merges, original regex) | 2.204 [2.180, 2.228] | 2.187 [2.161, 2.213] |
| **R1b** (R1_K32000 strings as added tokens) | **2.350 [2.324, 2.377]** | **2.188 [2.162, 2.213]** |
| R2_K32000 (merges, repaired regex) | 1.014 [1.004, 1.024] | 1.014 [1.004, 1.024] |

* **R2_K0**: the regex alone buys nothing. Llama gets slightly *worse* (+678 tokens, +1.0%); this matches TituLLM's 2.60. Qwen is unchanged (121,955 vs 121,958 tokens) because its vocabulary has almost no merges that span a mark. The whole R2 gain comes from regex plus merges together.
* **R1b**: 31,999 of the 32,000 R1 strings were added. One was dropped because it is a partial UTF-8 sequence that cannot be written as text. 7,336 (Llama) and 7,703 (Qwen) of them start with a space. R1b is never better than R1: Qwen comes out equal and Llama 0.15 worse.

**How added tokens bypass the regex.** HF `tokenizers` handles added tokens *before* normalisation and pre-tokenisation. A leftmost-longest Aho-Corasick scan over the raw string marks every occurrence of an added token's content and removes it as an atomic piece. Only the gaps between matches go through the Split regex and the byte-level BPE. So an added token can match across regex boundaries, for example across the mark cut that R1's `\p{L}+` imposes. It can also match inside a word and leave the stranded remainder to base BPE. A space can be orphaned too: the token `पह` matches without its preceding space, which leaves a lone `Ġ` token. This greedy string matching does not reproduce BPE's ranked merge order. It cannot exploit the base merges either. For Llama, which has many Devanagari merges, R1b is therefore worse than R1. For Qwen, which has few, the two are the same. Adding tokens does not get around the fragment floor in practice: the strings were learned on R1 fragments, so they rarely span a cut.

## 3. Mark-initial rate (Nepali devtest)

This is the share of tokens whose first character is a combining mark `\p{M}`. It is computed from each token's exact byte span (incremental byte decoding), so a token that starts inside a code point is never counted.

| Arm | Llama: Devanagari tokens | Llama: all tokens | Qwen: Devanagari tokens | Qwen: all tokens |
|---|---|---|---|---|
| R0 | 0.571 | 0.554 | 0.327 | 0.320 |
| R2_K0 | 0.568 | 0.549 | 0.327 | 0.320 |
| R1_K32000 | 0.668 | 0.649 | 0.668 | 0.643 |
| R1b | 0.653 | 0.611 | 0.668 | 0.643 |
| R2_K32000 | **0.147** | 0.135 | **0.120** | 0.108 |

R1 makes syllable breaking worse, while R2 cuts it to between a quarter and a third of R1 (W6). The regex alone (R2_K0) changes nothing, because the old vocabulary still contains the mark-initial merges. R1's rate is identical in both models (58,158 Devanagari tokens): at K=32k both reach the same fragment floor.

## 4. Unseen sequences (W4 note)

The reference set is every token id and adjacent id pair in the base tokenizer's segmentation of the first 30 MB of `data/ne_test_sample.txt` (5,130 lines). The rates below are shares of FLORES devtest Nepali token or bigram occurrences that are not in that set. "old-old" counts only bigrams where both ids come from the base vocabulary. R0 gives the sampling noise floor.

| Arm | Llama: tokens | Llama: bigrams | Llama: old-old bigrams (share of bigrams) | Qwen: tokens | Qwen: bigrams | Qwen: old-old bigrams (share) |
|---|---|---|---|---|---|---|
| R0 (floor) | 0.38% | 2.46% | 2.46% (100%) | 0.18% | 0.58% | 0.58% (100%) |
| R2_K0 | 0.39% | 3.36% | 3.36% (100%) | 0.18% | 0.58% | 0.58% (100%) |
| R1_K32000 | 15.5% | 30.4% | 3.23% (72%) | 72.4% | 93.2% | 11.4% (7.7%) |
| R2_K32000 | 74.2% | 93.5% | 31.1% (9.4%) | 84.1% | 95.9% | 27.0% (5.7%) |

For R2_K0, the "old tokens in contexts the model never saw" effect is small: +0.9 pp of bigrams for Llama and none for Qwen. Under R2_K32000, about 3 in 10 of the remaining old-old bigrams are new to the model, compared with a 0.6 to 2.5% floor. Those bigrams make up only 6 to 9% of all bigrams, though. Almost all of R2's novelty comes from the new tokens, which need new embeddings anyway.

## 5. Fragment inventory (W10)

These are distinct Devanagari-bearing pre-token types in `data/ne_test_sample.txt`. Llama and Qwen give similar counts; the Llama regex is shown.

| | Original regex (R1) | Repaired regex (R2) |
|---|---|---|
| Types, first 30 MB | 24,705 | 157,169 |
| Types, full 275 MB | 65,970 | 631,609 |
| Pre-tokens, full | 47.2M | 16.1M |
| Types covering 99% of pre-tokens | 5,892 | 470,596 |
| Types covering 99.9% | 29,724 | 615,508 |
| Mark-initial types | 39,609 (60%) | 540 |

Qwen gives 23,862 / 64,744 (original) and 156,326 / 630,383 (repaired). The 30 MB counts reproduce the smoke metas exactly. The 1 GB box metas give 124,596 (R1) and 1,536,573 (R2). Under R1, about 6k fragment types cover 99% of Nepali pre-tokens. Once K is a few thousand, almost every frequent fragment is a single token, so R1 is within 0.02 of its floor (2.17 to 2.19) by K=16k. The repaired regex leaves a long tail of whole words that merges can keep absorbing.

## 6. K-sweep build times (N7)

The 28 `meta.json` files are in `results/ksweep_meta/<model>/R{1,2}_K*/`. All runs used 1,000,000,463 bytes of merge text. **K=64,000 took 116.9 to 214.1 s**, and the whole sweep ranged from 93.4 to 214.1 s. That wall clock includes loading the tokenizer and pre-tokenising 1 GB. "Takes seconds" should read "takes 2 to 4 minutes on one CPU node". R1 is slower than R2 (166 to 214 s vs 93 to 148 s).
