#!/bin/bash
# Re-run Qwen R1 (OOM'd while an eval shared its GPU), evaluate it, then run Experiment B.
cd /home/ntt
log(){ echo "$(date -u +%H:%M:%S) $*"; }
until grep -q ALL_DONE logs/fast_all.log; do sleep 30; done
rm -rf runs/qwen_R1_s0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
venv/bin/python -m torch.distributed.run --nproc_per_node=4 train.py --model models/qwen_R1 --data data/qwen_R1 \
    --out runs/qwen_R1_s0 --lr 1e-4 --seed 0 --micro 4 --save_at 556 > logs/train_qwen_R1_s0.log 2>&1 \
    && log "TRAINED runs/qwen_R1_s0 RERUN" || log "TRAIN_FAIL runs/qwen_R1_s0 RERUN"
(CUDA_VISIBLE_DEVICES=0 venv/bin/python eval.py runs/qwen_R1_s0/model evals/qwen_R1_s0.json > logs/eval_qwen_R1_s0.log 2>&1;
 CUDA_VISIBLE_DEVICES=0 venv/bin/python eval.py runs/qwen_R1_s0/model_step556 evals/qwen_R1_s0_eqcompute.json --quick > logs/eval_qwen_R1_eq.log 2>&1) &
log CHAIN_QWEN_DONE
exec ./expB.sh
