import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="1"
from huggingface_hub import snapshot_download
L="hin_Deva mar_Deva ben_Beng guj_Gujr tam_Taml tel_Telu sin_Sinh tha_Thai mya_Mymr khm_Khmr amh_Ethi kor_Hang rus_Cyrl".split()
snapshot_download("HuggingFaceFW/fineweb-2", repo_type="dataset", allow_patterns=[f"data/{l}/train/000_00000.parquet" for l in L], local_dir="/home/ntt/raw/fw2")
print("DONE2")
