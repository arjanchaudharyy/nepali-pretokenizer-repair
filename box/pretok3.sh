#!/bin/bash
# Build 3 GB + 3 GB Llama token streams once the 3 GB corpora are ready.
cd /home/ntt
until [ -f corpus3/prep.npi_Deva.json ] && [ -f corpus3/prep.eng_Latn.json ]; do sleep 30; done
L=$(ls -d /home/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/*)/tokenizer.json
for arm in R0 R1 R2; do
  if [ $arm = R0 ]; then TOK=$L; else TOK=tok/Llama-3.2-1B/${arm}_K32000/tokenizer.json; fi
  CORPUS_DIR=/home/ntt/corpus3 venv/bin/python pretok3.py $TOK "<|end_of_text|>" 3e9 3e9 data3/llama_$arm > logs/pretok3_$arm.log 2>&1 &
done
wait; echo PRETOK3_DONE
