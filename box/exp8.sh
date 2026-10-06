#!/bin/bash
cd /home/ntt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
n=bos_llama_R2_s1
venv/bin/python -m torch.distributed.run --nproc_per_node=4 train.py --model models/llama_R2 --data data/llamabos_R2 --out runs/$n --lr 1e-4 --seed 1 --micro 4 > logs/train_$n.log 2>&1 && echo "$(date -u +%T) TRAINED $n" || echo "$(date -u +%T) TRAIN_FAIL $n"
CUDA_VISIBLE_DEVICES=0 venv/bin/python eval.py runs/$n/model evals/$n.json --bpb-only > logs/eval_$n.log 2>&1 && echo "$(date -u +%T) EVALED $n" || echo "$(date -u +%T) EVAL_FAIL $n"
echo "$(date -u +%T) EXP8_DONE"
