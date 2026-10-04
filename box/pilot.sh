#!/bin/bash
# LR pilot on R0: 200 steps per LR, pick by held-out native BPB (mean of ne and en).
cd /home/ntt
m=$1
for lr in 3e-5 1e-4 3e-4; do
  out=runs/pilot_${m}_$lr
  [ -f evals/pilot_${m}_$lr.json ] && continue
  venv/bin/python -m torch.distributed.run --nproc_per_node=4 train.py --model models/${m}_R0 --data data/${m}_R0 \
      --out $out --lr $lr --max_steps 200 --save 1 > logs/pilot_${m}_$lr.log 2>&1
  venv/bin/python eval.py $out/model evals/pilot_${m}_$lr.json --quick > logs/pilot_eval_${m}_$lr.log 2>&1
  rm -rf $out/model
done
venv/bin/python - <<PY
import json
r={lr:json.load(open(f"evals/pilot_${m}_{lr}.json")) for lr in ["3e-5","1e-4","3e-4"]}
for lr,v in r.items(): print(lr, round(v["bpb_native_ne"],4), round(v["bpb_native_en"],4))
best=min(r,key=lambda k:(r[k]["bpb_native_ne"]+r[k]["bpb_native_en"])/2)
print("BEST",best); open("evals/best_lr_${m}","w").write(best)
PY
