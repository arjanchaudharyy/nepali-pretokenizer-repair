#!/bin/bash
# cross-lingual K sweep (CPU, local). 9 langs x 2 bases x 2 arms, K in {4000,16000,32000}
cd "$(dirname "$0")/.."
export RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=true
jobs=()
for code in ben_Beng hin_Deva tam_Taml tha_Thai tel_Telu mya_Mymr kor_Hang sin_Sinh amh_Ethi; do
  for base in unsloth/Llama-3.2-1B Qwen/Qwen3-1.7B-Base; do
    for arm in R1 R2; do
      echo "$code $base $arm"
    done
  done
done | xargs -P ${PAR:-6} -L 1 bash -c 'code=$0; base=$1; arm=$2; short=$(basename $base); s=${code#*_};
  .venv/bin/python box/retrofit_tok_xl.py $base $arm 4000,16000,32000 $s data/xl/$code.txt 1e12 tok/xl/$code/$short > logs/xl/build_${code}_${short}_${arm}.log 2>&1; echo done $code $short $arm $?'
echo XL_BUILD_DONE
