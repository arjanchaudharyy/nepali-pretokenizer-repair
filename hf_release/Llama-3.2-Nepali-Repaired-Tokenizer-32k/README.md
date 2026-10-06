---
license: llama3.2
language:
- ne
- en
tags:
- tokenizer
- nepali
- devanagari
- vocabulary-extension
base_model: meta-llama/Llama-3.2-1B
---

# Llama-3.2-Nepali-Repaired-Tokenizer-32k

**Built with Llama.** A tokenizer for Llama-3.x models extended for Nepali, from the paper *Merge-Based Vocabulary Extension Stalls at the Pre-Tokenizer: An English-Preserving Repair for Nepali, Tested in Continued Pretraining* (Chaudhary, Naamche Labs, 2026). Code, data and pre-registration: https://github.com/arjanchaudharyy/nepali-pretokenizer-repair

## What it is

The pre-tokenizer regex is repaired so that combining marks (Devanagari vowel signs, virama, anusvara and so on) stay inside words, and 32,000 Nepali merges are appended by continued BPE. This is arm R2 in the paper.

Only the tokenizer is released here, not model weights. New token ids start at 128256; all original ids, including special tokens, are unchanged. To use it with a model, resize the model's embeddings to 160256 rows and initialise the new rows (the paper uses the mean of each new token's constituent rows; see `box/init_model.py` in the repository), then continue pretraining.

## Numbers (FLORES-200 devtest)

| | Nepali / English token ratio |
|---|---|
| Original tokenizer | 2.58x |
| This tokenizer | 1.01x |

English token ids are identical to the original tokenizer on all 1,012 English devtest sentences, and the tokenizer is lossless on the Nepali and English devtest text. The repair leaves the token ids of any input with no combining mark and no Devanagari character unchanged (Proposition 1 in the paper). Text in other scripts that uses combining marks (for example Hindi, Thai or vocalised Arabic) is tokenized differently.

## Loading

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Aarjan/Llama-3.2-Nepali-Repaired-Tokenizer-32k")
```

`tokenizer_config.json` sets `tokenizer_class` to `PreTrainedTokenizerFast` on purpose. Do not change it.

## Limitations

In the paper's small continued-pretraining experiments (1B and 0.6B models, 2 GB of text, one seed), extending the vocabulary of either kind gave worse held-out Nepali bits per byte than keeping the original vocabulary, and at equal bytes the repaired tokenizer scored worse on Nepali than standard extension, but it also trained for 26 percent fewer steps. Retrained for the same number of steps (two seeds each), it beat standard extension on Qwen3 (by 0.010 bits per byte, and also at a higher learning rate) and trailed it on Llama 3.2 (by 0.012). Keeping the original vocabulary stayed best throughout. It needs less than half as many decoding steps as standard extension on reference Nepali text, and about a third fewer on the models' own generations. Read the paper before relying on this tokenizer for a production model.

## Licence

This tokenizer is derived from Llama 3.2 and is distributed under the Llama 3.2 Community License (see LICENSE and USE_POLICY.md). Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved.

## Citation

```bibtex
@misc{chaudhary2026extend,
  title  = {Merge-Based Vocabulary Extension Stalls at the Pre-Tokenizer: An English-Preserving Repair for Nepali, Tested in Continued Pretraining},
  author = {Chaudhary, Aarjan},
  year   = {2026},
  note   = {Naamche Labs. Preprint.},
  url    = {https://github.com/arjanchaudharyy/nepali-pretokenizer-repair}
}
```
