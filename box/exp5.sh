#!/bin/bash
# Amendment 5: matched-steps R2 (Qwen, Llama-BOS) and Qwen seed 1. Same settings as the main runs.
cd /home/ntt
log(){ echo "$(date -u +%H:%M:%S) $*"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run(){ # name model data seed extra...
  n=$1; m=$2; d=$3; s=$4; shift 4
  venv/bin/python -m torch.distributed.run --nproc_per_node=4 train.py --model models/$m --data data/$d \
     --out runs/$n --lr 1e-4 --seed $s --micro 4 "$@" > logs/train_$n.log 2>&1 && log "TRAINED $n" || log "TRAIN_FAIL $n"
  (CUDA_VISIBLE_DEVICES=$G venv/bin/python eval.py runs/$n/model evals/$n.json --bpb-only > logs/eval_$n.log 2>&1 && log "EVALED $n" || log "EVAL_FAIL $n") &
  G=$(( (G+1) % 4 ))
}
G=0
run ms_qwen_R2_s0 qwen_R2 qwen_R2 0 --target_steps 748
run ms_bos_llama_R2_s0 llama_R2 llamabos_R2 0 --target_steps 737
run qwen_R2_s1 qwen_R2 qwen_R2 1
run qwen_R1_s1 qwen_R1 qwen_R1 1
wait
log EXP5_DONE
