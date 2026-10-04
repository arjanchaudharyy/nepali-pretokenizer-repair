#!/bin/bash
# Experiment B (PREREG_AMENDMENT3.md): embedding warm-up stage + full CPT; B1 = 1 GB+1 GB, B3 = 3 GB+3 GB.
cd /home/ntt
log(){ echo "$(date -u +%H:%M:%S) $*"; }
until grep -q ALL_DONE logs/fast_all.log; do sleep 60; done
log START_B
g=0
run_two_stage(){  # $1 model dir, $2 data dir, $3 out prefix
  local m=$1 d=$2 o=$3
  venv/bin/python -m torch.distributed.run --nproc_per_node=4 train.py --model $m --data $d --out ${o}_stage1 \
      --lr 1e-3 --seed 0 --frac 0.1 --only_embed 1 --micro 4 > logs/$(basename $o)_stage1.log 2>&1 || { log "TRAIN_FAIL ${o}_stage1"; return; }
  venv/bin/python -m torch.distributed.run --nproc_per_node=4 train.py --model ${o}_stage1/model --data $d --out $o \
      --lr 1e-4 --seed 0 --micro 4 > logs/$(basename $o).log 2>&1 || { log "TRAIN_FAIL $o"; return; }
  log "TRAINED $o $(venv/bin/python -c "import json;d=json.load(open('$o/summary.json'));print(d['steps'],d['tokens'],round(d['secs']))")"
  local G=$g; g=$(( (g+1) % 4 ))
  (CUDA_VISIBLE_DEVICES=$G venv/bin/python eval.py $o/model evals/$(basename $o).json > logs/eval_$(basename $o).log 2>&1; rm -rf ${o}_stage1/model) &
}
for arm in R0 R2 R1; do run_two_stage models/llama_$arm data/llama_$arm runs/B1_llama_$arm; done
for arm in R0 R2 R1; do [ -f data3/llama_$arm/meta.json ] && run_two_stage models/llama_$arm data3/llama_$arm runs/B3_llama_$arm; done
wait
log B_DONE
