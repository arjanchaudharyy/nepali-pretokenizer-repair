#!/bin/bash
# Amendment 6: second seeds for key arms, then a learning-rate check. Stops on the STOP file (budget cap).
cd /home/ntt
log(){ echo "$(date -u +%H:%M:%S) $*"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=0
run(){ # name model data seed lr extra...
  [ -f STOP ] && { log "STOPPED before $1"; return; }
  n=$1; m=$2; d=$3; s=$4; lr=$5; shift 5
  venv/bin/python -m torch.distributed.run --nproc_per_node=4 train.py --model models/$m --data data/$d \
     --out runs/$n --lr $lr --seed $s --micro 4 "$@" > logs/train_$n.log 2>&1 && log "TRAINED $n" || log "TRAIN_FAIL $n"
  (CUDA_VISIBLE_DEVICES=$G venv/bin/python eval.py runs/$n/model evals/$n.json --bpb-only > logs/eval_$n.log 2>&1 && log "EVALED $n" || log "EVAL_FAIL $n") &
  G=$(( (G+1) % 4 ))
}
run ms_qwen_R2_s1 qwen_R2 qwen_R2 1 1e-4 --target_steps 748
run ms_bos_llama_R2_s1 llama_R2 llamabos_R2 1 1e-4 --target_steps 737
run bos_llama_R1_s1 llama_R1 llamabos_R1 1 1e-4
run bos_llama_R0_s1 llama_R0 llamabos_R0 1 1e-4
run qwen_R0_s1 qwen_R0 qwen_R0 1 1e-4
run lr3_qwen_R1_s0 qwen_R1 qwen_R1 0 3e-4
run lr3_ms_qwen_R2_s0 qwen_R2 qwen_R2 0 3e-4 --target_steps 748
wait
log EXP6_DONE
