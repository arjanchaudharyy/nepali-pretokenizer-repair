# Merge-Based Vocabulary Extension Stalls at the Pre-Tokenizer: An English-Preserving Repair for Nepali, Tested in Continued Pretraining

Code and data for the paper by **Aarjan Chaudhary** (Naamche Labs).

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

## Released tokenizers (Hugging Face)

Exactly the K=32,000 tokenizers evaluated in the paper. Only tokenizers, no model weights.

| Tokenizer | Nepali / English (FLORES-200 devtest) |
|---|---|
| [Aarjan/Llama-3.2-Nepali-Repaired-Tokenizer-32k](https://huggingface.co/Aarjan/Llama-3.2-Nepali-Repaired-Tokenizer-32k) (R2, Built with Llama) | 1.01x (original 2.58x) |
| [Aarjan/Llama-3.2-Nepali-Extended-Tokenizer-32k](https://huggingface.co/Aarjan/Llama-3.2-Nepali-Extended-Tokenizer-32k) (R1 baseline, Built with Llama) | 2.20x |
| [Aarjan/Qwen3-Nepali-Repaired-Tokenizer-32k](https://huggingface.co/Aarjan/Qwen3-Nepali-Repaired-Tokenizer-32k) (R2) | 1.01x (original 4.42x) |
| [Aarjan/Qwen3-Nepali-Extended-Tokenizer-32k](https://huggingface.co/Aarjan/Qwen3-Nepali-Extended-Tokenizer-32k) (R1 baseline) | 2.19x |

**Pitfall when repairing a Qwen tokenizer.** In transformers 5.x, the `Qwen2Tokenizer` class rebuilds the
pre-tokenizer from a hard-coded pattern when it loads, which silently undoes a regex repair stored in
`tokenizer.json`. Our releases set `tokenizer_class` to `PreTrainedTokenizerFast` to avoid this.
`hf_release/build.py` rebuilds and re-verifies all four (English ids identical, lossless, special-token ids unchanged).

## Repository map

| Path | What |
|---|---|
| `floorcheck.py` | One-command floor diagnostic |
| `measure.py`, `tok_registry.py`, `audit.py` | Production audit of 33 tokenizers on 36 FLORES languages |
| `ceiling.py` | Pre-token floors |
| `released_extensions.py` | Released vocabulary extensions vs their floors |
| `tokenizer_extras.py` | Regex-only and added-token arms, syllable-break rates |
| `verify_retrofit.py` | Checks extended tokenizers: lossless decoding, English ids unchanged |
| `analyze_cpt.py` | Continued-pretraining analysis (paired bootstrap) |
| `results/revision/*.py` | Later analyses: clustered bootstrap, equal compute, matched steps, seeds, learning rate, two-tokenization scoring, full-pipeline floors |
| `make_*.py`, `fig_*.py` | Every table and figure in the paper, generated from `results/` |
| `box/` | GPU-box pipeline: data, tokenizers, training, evaluation (see `box/README.md`) |
| `hf_release/` | Builder and model cards for the released tokenizers |
| `PREREG*.md` | Analysis plan and amendments; the paper's appendix maps file names to its numbering and lists every change made after seeing results |
| `results/` | All measured numbers, per-piece scores and training logs |
| `logs/` | Logs of the tokenizer sweeps |
| `paper/` | LaTeX source (`make_arxiv.sh` builds the PDFs and the arXiv bundle) |
| `arxiv_preview.pdf` | The paper |

## Reproduce

From a fresh clone (CPU; only tokenizer files and public data are downloaded):
```bash
pip install -r requirements.txt
curl -sL https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz | tar xz
python floorcheck.py unsloth/Llama-3.2-1B npi_Deva   # one-command floor diagnostic
python ceiling.py                                    # pre-token floors
python released_extensions.py                        # released extensions vs their floors
python make_decomp_table.py                          # premium decomposition (full pre-tokenizer pipeline)
python measure.py results/sweep_devtest.json         # the 33-tokenizer audit (slow; some repositories are gated)
python fetch_ne_test_sample.py                       # FineWeb-2 npi_Deva test split -> data/ne_test_sample.txt
python results/revision/amendment6.py                # every continued-pretraining comparison, from saved per-piece scores
```
`data/ne_test_sample.txt` is the FineWeb-2 `npi_Deva` test split (ODC-By), one document per line; `fetch_ne_test_sample.py` rebuilds it byte for byte.

Needs files from the GPU box (`tok/box/`, not in the repository): `tokenizer_extras.py`, `results/revision/devanagari_only.py` and `hf_release/build.py`, which read the extended tokenizers produced by `box/retrofit_tok.py`. The released tokenizers themselves are on Hugging Face.

The K-sweep, cross-script sweep and continued pretraining ran on a 4x H200 box (`box/`, see `box/README.md`). Those scripts assume the box layout (`/home/ntt`) and FineWeb-2 / FineWeb-Edu downloads. Every run's evaluation, per-piece scores and training log is in `results/cpt/`, and all statistics and tables in the paper regenerate from them on CPU (`results/revision/*.py`, `make_*_table.py`). Trained checkpoints are not released.

The 503 held-out Nepali document boundaries used for the clustered intervals are in `results/revision/heldout_ne_doc_ids.json`.

## Licences

Code: MIT. Tokenizers derived from Llama 3.2 follow the Llama 3.2 Community License
("Built with Llama"); derived from Qwen3: Apache-2.0. Data: FineWeb-2 and FineWeb-Edu
(ODC-By), FLORES-200 and Belebele (CC BY-SA 4.0).

## Use of AI assistance

This work was carried out with substantial assistance from a large language model assistant
under the author's direction; see the paper's ethics statement.
