#!/bin/bash
# Corrected fast pipeline. Llama-3.2-1B and Qwen3-0.6B-Base; arms R0/R1/R2; K=32000;
# 1 GB Nepali + 1 GB English per arm (identical documents); lr 1e-4; seed 0.
set -u
cd /home/ntt
log(){ echo "$(date -u +%H:%M:%S) $*"; }
rm -f corpus/*.txt corpus/*.jsonl
venv/bin/python prep.py npi_Deva 1.0e9 3e7 > logs/prep_ne.log 2>&1 &
venv/bin/python prep.py eng_Latn 1.0e9 3e7 > logs/prep_en.log 2>&1 &
wait; log PREP_DONE; cat logs/prep.npi_Deva.json logs/prep.eng_Latn.json | tr -d '\n'; echo
declare -A REPO=( [llama]=unsloth/Llama-3.2-1B [qwen]=Qwen/Qwen3-0.6B-Base )
declare -A TOKD=( [llama]=Llama-3.2-1B [qwen]=Qwen3-1.7B-Base )   # Qwen3 tokenizers are identical across sizes
declare -A EOS=( [llama]="<|end_of_text|>" [qwen]="<|endoftext|>" )
rm -rf data/llama_* data/qwen_* models/llama_* models/qwen_* runs/llama_* runs/qwen_* evals/*.json
for m in llama qwen; do
  for arm in R0 R1 R2; do
    if [ $arm = R0 ]; then EXT=none; TOK=$(venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('${REPO[$m]}', allow_patterns=['*.json','*.safetensors','*.txt']))")/tokenizer.json
    else EXT=tok/${TOKD[$m]}/${arm}_K32000; TOK=$EXT/tokenizer.json; fi
    venv/bin/python pretok.py $TOK "${EOS[$m]}" 1e9 1e9 data/${m}_$arm > logs/pretok_${m}_$arm.log 2>&1 &
    venv/bin/python init_model.py ${REPO[$m]} $EXT models/${m}_$arm > logs/init_${m}_$arm.log 2>&1 &
  done
done
wait; log BUILD_DONE
grep -h '^{' logs/init_*.log logs/pretok_*.log | cut -c1-260
for m in llama qwen; do for arm in R0 R1 R2; do [ -f models/${m}_$arm/config.json ] && [ -f data/${m}_$arm/meta.json ] || { log "MISSING ${m}_$arm"; exit 1; }; done; done
# base-model evals on GPU 3 while training runs on all 4
(CUDA_VISIBLE_DEVICES=3 venv/bin/python eval.py models/llama_R0 evals/llama_base.json > logs/eval_llama_base.log 2>&1; \
 CUDA_VISIBLE_DEVICES=3 venv/bin/python eval.py models/qwen_R0 evals/qwen_base.json > logs/eval_qwen_base.log 2>&1) &
g=0
for m in llama qwen; do
  R2STEPS=""
  for arm in R2 R0 R1; do
    out=runs/${m}_${arm}_s0
    extra=""; [ $arm != R2 ] && extra="--save_at $R2STEPS"
    venv/bin/python -m torch.distributed.run --nproc_per_node=4 train.py --model models/${m}_$arm --data data/${m}_$arm \
        --out $out --lr 1e-4 --seed 0 $extra > logs/train_${m}_${arm}_s0.log 2>&1 || log "TRAIN_FAIL $out"
    [ $arm = R2 ] && R2STEPS=$(venv/bin/python -c "import json;print(json.load(open('$out/summary.json'))['steps'])")
    log "TRAINED $out $(venv/bin/python -c "import json;d=json.load(open('$out/summary.json'));print(d['steps'],d['tokens'],round(d['secs']),round(d['final_loss'],4))")"
    G=$g; g=$(( (g+1) % 3 ))
    (CUDA_VISIBLE_DEVICES=$G venv/bin/python eval.py $out/model evals/${m}_${arm}_s0.json > logs/eval_${m}_${arm}_s0.log 2>&1
     [ -d $out/model_step$R2STEPS ] && CUDA_VISIBLE_DEVICES=$G venv/bin/python eval.py $out/model_step$R2STEPS evals/${m}_${arm}_s0_eqcompute.json --quick > logs/eval_${m}_${arm}_eq.log 2>&1) &
  done
done
wait
log ALL_DONE
