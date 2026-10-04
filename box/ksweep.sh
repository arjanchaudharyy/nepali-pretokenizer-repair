#!/bin/bash
# vocabulary-size sweep for both base models and both arms (compression only)
cd /home/ntt
export RAYON_NUM_THREADS=8
venv/bin/python -c "import sys
n=0
with open(\"corpus/npi_Deva.txt\",encoding=\"utf-8\") as f, open(\"tok/ne_1gb.txt\",\"w\",encoding=\"utf-8\") as o:
  for l in f:
    o.write(l); n+=len(l.encode())
    if n>=1e9: break"
for base in unsloth/Llama-3.2-1B Qwen/Qwen3-1.7B-Base; do
  short=$(basename $base)
  for K in 1000 2000 4000 8000 16000 32000 64000; do
    for arm in R1 R2; do
      venv/bin/python retrofit_tok.py $base $arm $K tok/ne_1gb.txt 1e9 tok/$short/${arm}_K$K > logs/ksweep_${short}_${arm}_K$K.log 2>&1 &
    done
  done
  wait
done
venv/bin/python ksweep_eval.py > logs/ksweep_eval.log 2>&1
echo KSWEEP_DONE
