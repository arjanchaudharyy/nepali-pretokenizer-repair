import os; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="1"
from huggingface_hub import snapshot_download
for r in ["unsloth/Llama-3.2-1B","Qwen/Qwen3-1.7B-Base"]:
    print(snapshot_download(r, allow_patterns=["*.json","*.safetensors","*.txt","*.model"]))
