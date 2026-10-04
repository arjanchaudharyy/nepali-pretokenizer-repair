#!/bin/bash
# Diagnostic re-scoring with the end-of-text separator as document prefix (see BOS artifact).
cd /home/ntt; mkdir -p evals_eos
run(){ [ -f evals_eos/$2.json ] || EVAL_PREFIX=eos CUDA_VISIBLE_DEVICES=3 venv/bin/python eval.py $1 evals_eos/$2.json --bpb-only > logs/evaleos_$2.log 2>&1; }
run models/llama_R0 llama_base; run models/qwen_R0 qwen_base
for m in llama qwen; do for a in R0 R1 R2; do run runs/${m}_${a}_s0/model ${m}_${a}_s0; done; done
for a in R0 R2 R1; do
  until [ -f evals/B1_llama_$a.json ] || grep -q "B_DONE" logs/chain.log; do sleep 60; done
  [ -d runs/B1_llama_$a/model ] && run runs/B1_llama_$a/model B1_llama_$a
done
echo EOS_DONE
