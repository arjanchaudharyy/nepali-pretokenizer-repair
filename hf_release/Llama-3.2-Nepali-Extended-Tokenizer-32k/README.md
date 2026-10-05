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

# Llama-3.2-Nepali-Extended-Tokenizer-32k

**Built with Llama.** A tokenizer for Llama-3.x models extended for Nepali, from the paper *Extend or Repair? Vocabulary Extension Cannot Cross a Pre-Tokenizer Boundary* (Chaudhary and Dhakal, Naamche Labs, 2026). Code, data and pre-registration: https://github.com/arjanchaudharyy/nepali-pretokenizer-repair

## What it is

The original letters-only pre-tokenizer regex is kept and 32,000 Nepali merges are appended by continued BPE (the standard vocabulary-extension recipe). This is arm R1 in the paper, released as a baseline; it is limited by the pre-token floor described in the paper.

Only the tokenizer is released here, not model weights. New token ids start at 128256; all original ids, including special tokens, are unchanged. To use it with a model, resize the model's embeddings to 160256 rows and initialise the new rows (the paper uses the mean of each new token's constituent rows; see `box/init_model.py` in the repository), then continue pretraining.

## Numbers (FLORES-200 devtest)

| | Nepali / English token ratio |
|---|---|
| Original tokenizer | 2.58x |
| This tokenizer | 2.20x |

English token ids are identical to the original tokenizer on all 1,012 English devtest sentences, and the tokenizer is lossless on the Nepali and English devtest text.

## Loading

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Aarjan/Llama-3.2-Nepali-Extended-Tokenizer-32k")
```

`tokenizer_config.json` sets `tokenizer_class` to `PreTrainedTokenizerFast` on purpose. Do not change it.

## Limitations

In the paper's small continued-pretraining experiments (1B and 0.6B models, 2 GB of text, one seed), extending the vocabulary of either kind gave worse held-out Nepali bits per byte than keeping the original vocabulary, and the repaired tokenizer was 2 to 6 percent worse on Nepali than standard extension while needing 26 percent fewer training tokens. Read the paper before relying on this tokenizer for a production model.

## Licence

This tokenizer is derived from Llama 3.2 and is distributed under the Llama 3.2 Community License (see LICENSE and USE_POLICY.md). Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved.

## Citation

```bibtex
@misc{chaudhary2026extend,
  title  = {Extend or Repair? Vocabulary Extension Cannot Cross a Pre-Tokenizer Boundary},
  author = {Chaudhary, Aarjan and Dhakal, Navyata},
  year   = {2026},
  note   = {Naamche Labs. Preprint.},
  url    = {https://github.com/arjanchaudharyy/nepali-pretokenizer-repair}
}
```
