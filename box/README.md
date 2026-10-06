# GPU-box pipeline

These scripts ran on a 4x H200 machine with the repository at `/home/ntt`. Paths assume that layout.

| Script | Role |
|---|---|
| `setup.sh`, `dl.py`, `fetch.py`, `fetch2.py` | Environment and downloads (FineWeb-2 `npi_Deva`, FineWeb-Edu, FLORES-200, base models) |
| `prep.py`, `prep_xl.py`, `decontam_char_xl.py` | Cleaning, deduplication, FLORES 10-gram decontamination, held-out slices |
| `retrofit_tok.py`, `retrofit_tok_xl.py` | Continued-BPE extension under the original (R1) or repaired (R2) regex |
| `ksweep.sh`, `ksweep_eval.py`, `xl_sweep.sh`, `xl_eval.py` | Merge-count sweep and cross-script sweep |
| `init_model.py` | Embedding resize and mean initialisation |
| `pretok.py` | Token streams for each arm (optional BOS per document) |
| `train.py` | Continued pretraining (DDP); `--target_steps` for matched-steps runs |
| `eval.py` | Byte-normalised evaluation; `EVAL_PREFIX=eos` scores after the end-of-text token |
| `marg.py` | Two-tokenization scoring (fourth amendment) |
| `chk_bos.py` | BOS diagnostic reported in the appendix |
| `fast_all.sh` | Main runs (both models, three arms) |
| `chain.sh`, `expB.sh` | Qwen R1 rerun after an out-of-memory failure; warm-up runs (second amendment) |
| `eval_eos.sh` | Rescoring after the training separator |
| `expC.sh` | BOS-consistent Llama rerun (third amendment) |
| `exp5.sh` to `exp8.sh` | Matched-steps, second-seed and learning-rate runs (fourth to seventh amendments) |
