#!/bin/bash
# Fast plan: lr 1e-4, first 1/3 of each stream, seed 0 for all arms, then seed 1 if asked.
cd /home/ntt
m=$1; seeds=${2:-0}
for seed in $seeds; do
  for arm in R0 R2 R1; do
    out=runs/${m}_${arm}_s$seed
    [ -f $out/summary.json ] && continue
    venv/bin/python -m torch.distributed.run --nproc_per_node=4 train.py --model models/${m}_$arm --data data/${m}_$arm \
        --out $out --lr 1e-4 --seed $seed --frac 0.3333 > logs/train_${m}_${arm}_s$seed.log 2>&1 || echo "TRAIN_FAIL $out"
    echo "TRAINED $out $(grep -o "'tokens': [0-9]*" logs/train_${m}_${arm}_s$seed.log | tail -1)"
    CUDA_VISIBLE_DEVICES=0 venv/bin/python eval.py $out/model evals/${m}_${arm}_s$seed.json > logs/eval_${m}_${arm}_s$seed.log 2>&1 &
  done
done
wait
echo "QUEUE_DONE $m"
