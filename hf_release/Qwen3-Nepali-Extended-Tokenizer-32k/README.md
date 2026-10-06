---
license: apache-2.0
language:
- ne
- en
tags:
- tokenizer
- nepali
- devanagari
- vocabulary-extension
base_model: Qwen/Qwen3-1.7B-Base
---

# Qwen3-Nepali-Extended-Tokenizer-32k

A tokenizer for Qwen3 models extended for Nepali, from the paper *Merge-Based Vocabulary Extension Stalls at the Pre-Tokenizer: An English-Preserving Repair for Nepali, Tested in Continued Pretraining* (Chaudhary, Naamche Labs, 2026). Code, data and pre-registration: https://github.com/arjanchaudharyy/nepali-pretokenizer-repair

## What it is

The original letters-only pre-tokenizer regex is kept and 32,000 Nepali merges are appended by continued BPE (the standard vocabulary-extension recipe). This is arm R1 in the paper, released as a baseline; it is limited by the pre-token floor described in the paper.

Only the tokenizer is released here, not model weights. New token ids start at 151669; all original ids, including special tokens, are unchanged. To use it with a model, resize the model's embeddings to 183669 rows and initialise the new rows (the paper uses the mean of each new token's constituent rows; see `box/init_model.py` in the repository), then continue pretraining.

## Numbers (FLORES-200 devtest)

| | Nepali / English token ratio |
|---|---|
| Original tokenizer | 4.42x |
| This tokenizer | 2.19x |

English token ids are identical to the original tokenizer on all 1,012 English devtest sentences, and the tokenizer is lossless on the Nepali and English devtest text.

## Loading

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Aarjan/Qwen3-Nepali-Extended-Tokenizer-32k")
```

`tokenizer_config.json` sets `tokenizer_class` to `PreTrainedTokenizerFast` on purpose. Model-specific classes such as transformers' `Qwen2Tokenizer` rebuild the pre-tokenizer from a hard-coded pattern and would silently undo the repair. Do not change it.

## Limitations

In the paper's small continued-pretraining experiments (1B and 0.6B models, 2 GB of text, one or two seeds per arm), extending the vocabulary of either kind gave worse held-out Nepali bits per byte than keeping the original vocabulary, and at equal bytes the repaired tokenizer scored worse on Nepali than standard extension, but it also trained for 26 percent fewer steps. Retrained for the same number of steps (two seeds each), it beat standard extension on Qwen3 (by 0.010 bits per byte, and also at a higher learning rate) and trailed it on Llama 3.2 (by 0.012). Keeping the original vocabulary stayed best throughout. It needs less than half as many decoding steps as standard extension on reference Nepali text, and about a third fewer on the models' own generations. Read the paper before relying on this tokenizer for a production model.

## Licence

Apache-2.0, as for the Qwen3 base tokenizer (see LICENSE).

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
