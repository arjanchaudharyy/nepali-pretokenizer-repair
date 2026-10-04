#!/bin/bash
# Amendment 4: BOS-consistent Llama rerun. Same documents/order/settings as the main runs, plus BOS at document starts.
cd /home/ntt
log(){ echo "$(date -u +%H:%M:%S) $*"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$(ls -d /home/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/*)/tokenizer.json
for a in R0 R1 R2; do
  if [ $a = R0 ]; then T=$L; else T=tok/Llama-3.2-1B/${a}_K32000/tokenizer.json; fi
  venv/bin/python pretok.py $T "<|end_of_text|>" 1e9 1e9 data/llamabos_$a "<|begin_of_text|>" > logs/pretok_bos_$a.log 2>&1 &
done
wait; log PRETOK_DONE; grep -h "^{" logs/pretok_bos_*.log | cut -c1-220
g=0
for a in R2 R0 R1; do
  o=runs/bos_llama_${a}_s0
  venv/bin/python -m torch.distributed.run --nproc_per_node=4 train.py --model models/llama_$a --data data/llamabos_$a \
     --out $o --lr 1e-4 --seed 0 --micro 4 > logs/train_bos_$a.log 2>&1 && log "TRAINED $o" || log "TRAIN_FAIL $o"
  G=$g; g=$(( (g+1) % 4 ))
  (CUDA_VISIBLE_DEVICES=$G venv/bin/python eval.py $o/model evals/bos_llama_${a}_s0.json --bpb-only > logs/eval_bos_$a.log 2>&1;
   EVAL_PREFIX=eos CUDA_VISIBLE_DEVICES=$G venv/bin/python eval.py $o/model evals_eos/bos_llama_${a}_s0.json --bpb-only > logs/evaleos_bos_$a.log 2>&1) &
done
wait
log C_DONE
