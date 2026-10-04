import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="1"
from huggingface_hub import snapshot_download
snapshot_download("HuggingFaceFW/fineweb-2", repo_type="dataset", allow_patterns=["data/npi_Deva/*"], local_dir="/home/ntt/raw/fw2")
snapshot_download("HuggingFaceFW/fineweb-edu", repo_type="dataset", allow_patterns=["sample/10BT/00[0-5]_*"], local_dir="/home/ntt/raw/fwedu")
print("DONE")
