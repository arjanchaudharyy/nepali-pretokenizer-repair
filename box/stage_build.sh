#!/bin/bash
# Stage B+C: initialise models and build token streams for every (model, arm).
# Usage: stage_build.sh <K> <ne_bytes> <en_bytes>
set -e
cd /home/ntt
K=$1; NE=$2; EN=$3
declare -A REPO=( [llama]=unsloth/Llama-3.2-1B [qwen]=Qwen/Qwen3-1.7B-Base )
declare -A SHORT=( [llama]=Llama-3.2-1B [qwen]=Qwen3-1.7B-Base )
declare -A EOS=( [llama]="<|end_of_text|>" [qwen]="<|endoftext|>" )
for m in ${MODELS:-llama qwen}; do
  for arm in R0 R1 R2; do
    if [ $arm = R0 ]; then EXT=none; TOK=$(ls -d /home/.cache/huggingface/hub/models--${REPO[$m]/\//--}/snapshots/*)/tokenizer.json
    else EXT=tok/${SHORT[$m]}/${arm}_K$K; TOK=$EXT/tokenizer.json; fi
    [ -f models/${m}_$arm/config.json ] || venv/bin/python init_model.py ${REPO[$m]} $EXT models/${m}_$arm > logs/init_${m}_$arm.log 2>&1
    [ -f data/${m}_$arm/meta.json ] || venv/bin/python pretok.py $TOK "${EOS[$m]}" $NE $EN data/${m}_$arm > logs/pretok_${m}_$arm.log 2>&1 &
  done
done
wait
echo BUILD_DONE
