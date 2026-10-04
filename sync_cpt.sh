#!/bin/bash
# Pull CPT evals and run summaries from the box, then run the pre-registered analysis.
cd "$(dirname "$0")"
mkdir -p results/cpt/evals results/cpt/runs
rsync -q -r -e ssh ntt:/home/ntt/evals/ results/cpt/evals/ < /dev/null
for d in $(ssh ntt 'ls -d /home/ntt/runs/llama_* /home/ntt/runs/qwen_* 2>/dev/null' < /dev/null); do
  n=$(basename $d); mkdir -p results/cpt/runs/$n
  rsync -q -e ssh "ntt:$d/summary.json" "ntt:$d/train_log.jsonl" results/cpt/runs/$n/ < /dev/null 2>/dev/null
done
.venv/bin/python analyze_cpt.py && .venv/bin/python fig_cpt.py
