"""Pre-token floor using each tokenizer's full Hugging Face pre-tokenizer pipeline, not only its last regex
(audit.py). Differs from audit.py only where the pipeline has extra steps (Falcon3: Digits after ByteLevel)."""
import json, unicodedata
from transformers import AutoTokenizer
L = lambda c: [unicodedata.normalize("NFC", x) for x in open(f"flores200_dataset/devtest/{c}.devtest").read().splitlines()]
ne, en = L("npi_Deva"), L("eng_Latn")
aud = {r["name"]: r for r in json.load(open("results/audit_unique.json"))}
REPOS = {"Llama-3": "NousResearch/Meta-Llama-3-8B", "Llama-4": "unsloth/Llama-4-Scout-17B-16E-Instruct",
         "Qwen2.5": "Qwen/Qwen2.5-7B", "DeepSeek-V3": "deepseek-ai/DeepSeek-V3", "Mistral-NeMo": "unsloth/Mistral-Nemo-Base-2407",
         "GLM-4.5": "zai-org/GLM-4.5", "GLM-5.3": "zai-org/GLM-5.3", "Falcon3": "tiiuae/Falcon3-7B-Base",
         "Arkios": "sajalregmi4/arkios-tokenizer", "Qwen3.5": "Qwen/Qwen3.5-9B", "LFM2": "LiquidAI/LFM2-8B-A1B",
         "TituLLM": "hishab/titulm-llama-3.2-1b-v2.0"}
out = {}
for n, repo in REPOS.items():
    t = AutoTokenizer.from_pretrained(repo)
    E = sum(len(t(s, add_special_tokens=False)["input_ids"]) for s in en)
    P = sum(len(t.backend_tokenizer.pre_tokenizer.pre_tokenize_str(s)) for s in ne)
    out[n] = dict(full_pipeline_floor=P / E, audit_regex_floor=aud[n]["floor"])
    print(f"{n:<13} full={P/E:.3f} audit={aud[n]['floor']:.3f}")
json.dump(out, open("results/revision/full_pipeline_floor.json", "w"), indent=1)
