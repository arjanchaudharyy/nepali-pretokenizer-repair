#!/bin/bash
set -e
cd /home/ntt
python -m venv --system-site-packages venv
venv/bin/pip install -q pyarrow tokenizers tiktoken regex transformers datasets huggingface_hub hf_transfer accelerate sacrebleu liger-kernel
tar xzf flores200.tar.gz
venv/bin/python fetch.py
venv/bin/python fetch2.py
venv/bin/python dl.py
echo SETUP_DONE
venv/bin/python prep.py npi_Deva 1e12 3e7 > logs/prep_ne.log 2>&1 &
venv/bin/python prep.py eng_Latn 3e10 3e7 > logs/prep_en.log 2>&1 &
wait
echo PREP_DONE
