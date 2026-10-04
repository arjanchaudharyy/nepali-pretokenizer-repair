#!/bin/bash
# Full pipeline for one model after stage_build: LR pilot -> 3 arms x 2 seeds -> evals.
cd /home/ntt
m=$1
./pilot.sh $m > logs/pilot_$m.log 2>&1
LR=$(cat evals/best_lr_$m)
echo "LR=$LR"
for seed in 0 1; do
  for arm in R0 R1 R2; do
    out=runs/${m}_${arm}_s$seed
    [ -f $out/summary.json ] && continue
    venv/bin/python -m torch.distributed.run --nproc_per_node=4 train.py --model models/${m}_$arm --data data/${m}_$arm \
        --out $out --lr $LR --seed $seed > logs/train_${m}_${arm}_s$seed.log 2>&1 || echo "TRAIN_FAIL $out"
    echo "TRAINED $out $(tail -1 logs/train_${m}_${arm}_s$seed.log | cut -c1-200)"
  done
done
# evals: base model, the two resized-but-untrained models, and all six CPT checkpoints, 4 at a time
jobs_list="models/${m}_R0:base models/${m}_R1:init_R1 models/${m}_R2:init_R2"
for seed in 0 1; do for arm in R0 R1 R2; do jobs_list="$jobs_list runs/${m}_${arm}_s$seed/model:${arm}_s$seed"; done; done
g=0
for j in $jobs_list; do
  d=${j%%:*}; tag=${j##*:}
  [ -f evals/${m}_$tag.json ] && continue
  CUDA_VISIBLE_DEVICES=$g venv/bin/python eval.py $d evals/${m}_$tag.json > logs/eval_${m}_$tag.log 2>&1 &
  g=$(( (g+1) % 4 )); [ $g -eq 0 ] && wait
done
wait
echo "QUEUE_DONE $m"
