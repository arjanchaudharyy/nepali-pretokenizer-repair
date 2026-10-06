#!/bin/bash
cd /home/ntt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
n=lr3_qwen_R0_s0
venv/bin/python -m torch.distributed.run --nproc_per_node=4 train.py --model models/qwen_R0 --data data/qwen_R0 --out runs/$n --lr 3e-4 --seed 0 --micro 4 > logs/train_$n.log 2>&1 && echo "$(date -u +%T) TRAINED $n" || echo "$(date -u +%T) TRAIN_FAIL $n"
CUDA_VISIBLE_DEVICES=0 venv/bin/python eval.py runs/$n/model evals/$n.json --bpb-only > logs/eval_$n.log 2>&1 && echo "$(date -u +%T) EVALED $n" || echo "$(date -u +%T) EVAL_FAIL $n"
echo "$(date -u +%T) EXP7_DONE"
