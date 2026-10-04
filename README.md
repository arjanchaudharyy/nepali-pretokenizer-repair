# Extend or Repair? Vocabulary Extension Cannot Cross a Pre-Tokenizer Boundary

Code and data for the paper by **Aarjan Chaudhary, Shrey Sharma and Navyata Dhakal**.

Many deployed open LLMs (Llama-3.x, Qwen2.5/3, GLM, Phi-4, OLMo-2, Granite, LFM2)
pre-tokenize text with a regex whose word class is `\p{L}+`. Vowel signs in
Devanagari and other Indic scripts are Unicode *combining marks* (`\p{M}`), so the
regex cuts every word apart before BPE runs. Vocabulary extension cannot cross
those cuts: tokens can never fall below the number of pre-tokens. This repository
measures that floor, shows that released extensions already sit on it, and
provides a two-substitution repair that leaves the token ids of every input
without combining marks or Devanagari (all ASCII English and code) unchanged.

## Check your own model in one command

```bash
python floorcheck.py unsloth/Llama-3.2-1B npi_Deva
```
```
regex class      letters-only (\p{L}+)
tokens           70,188
floor            59,578   (no extension under this regex can go below)
tokens / floor   1.178
repaired floor   21,530   (extra room from the repair: 2.77x)
premium          2.58x English (FLORES-200 devtest)
```
Works with any Hugging Face repo or `tokenizer.json`, and any FLORES-200 code or text
file. Only tokenizer files are downloaded; no weights, no remote code.

## Repository map

| Path | What |
|---|---|
| `floorcheck.py` | One-command floor diagnostic |
| `measure.py`, `tok_registry.py`, `audit.py` | Production audit of 33 tokenizers on 36 FLORES languages |
| `ceiling.py` | Pre-token floors |
| `released_extensions.py` | 12 released vocabulary extensions vs their floors |
| `box/retrofit_tok.py` | Repair + continued-BPE extension (R1, R2) |
| `box/retrofit_tok_xl.py`, `box/xl_*.py` | Cross-script sweep (9 languages) |
| `tokenizer_extras.py` | Regex-only and added-token arms, syllable-break rates |
| `box/prep.py`, `box/pretok.py` | Corpus cleaning (dedup, FLORES 10-gram decontamination), token streams |
| `box/init_model.py` | Embedding resize and mean initialisation, with exactness check |
| `box/train.py` | Continued pretraining (DDP), optional embedding-only warm-up |
| `box/eval.py` | Per-byte evaluation: BPB, Belebele, FLORES chrF++, generation speed |
| `analyze_cpt.py` | Pre-registered paired-bootstrap analysis |
| `PREREG*.md` | Pre-registration and amendments (see git history for timestamps) |
| `results/` | All measured numbers as JSON |
| `paper/` | LaTeX source |

## Reproduce

Tokenizer-side results (CPU, minutes):
```bash
python measure.py results/sweep_devtest.json   # production audit
python ceiling.py
python released_extensions.py
python tokenizer_extras.py
```
Continued-pretraining results need GPUs (we used 4x H200; see `box/fast_all.sh`).

FLORES-200: `curl -sL https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz | tar xz`

## Licences

Code: MIT. Tokenizers derived from Llama 3.2 follow the Llama 3.2 Community License
("Built with Llama"); derived from Qwen3: Apache-2.0. Data: FineWeb-2 and FineWeb-Edu
(ODC-By), FLORES-200 and Belebele (CC BY-SA 4.0).

## Use of AI assistance

This work was carried out with substantial assistance from a large language model assistant under the
authors' direction; see the paper's ethics statement.
