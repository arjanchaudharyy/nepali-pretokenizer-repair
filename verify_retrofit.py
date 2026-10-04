import sys, json, unicodedata
from pathlib import Path
from tokenizers import Tokenizer
from transformers import AutoTokenizer
FL=Path("flores200_dataset")
def load(c,s="devtest"): return [unicodedata.normalize("NFC",x) for x in (FL/s/f"{c}.{s}").read_text().splitlines()]
base_repo=sys.argv[1]; dirs=sys.argv[2:]
base=AutoTokenizer.from_pretrained(base_repo).backend_tokenizer
en,ne=load("eng_Latn"),load("npi_Deva")
code_en=open("measure.py").read()  # English code/comments: punctuation-heavy stress text
rus,hin,amh=load("rus_Cyrl"),load("hin_Deva"),load("amh_Ethi")
E0=sum(len(base.encode(s,add_special_tokens=False).ids) for s in en)
N0=sum(len(base.encode(s,add_special_tokens=False).ids) for s in ne)
print(f"base  ne/en={N0/E0:.3f}")
for d in dirs:
    tk=Tokenizer.from_file(f"{d}/tokenizer.json")
    same=lambda xs: all(base.encode(s,add_special_tokens=False).ids==tk.encode(s,add_special_tokens=False).ids for s in xs)
    N=sum(len(tk.encode(s,add_special_tokens=False).ids) for s in ne)
    lossless=all(tk.decode(tk.encode(s,add_special_tokens=False).ids)==s for s in ne+en)
    print(f"{d:28} ne/en={N/E0:.3f}  english_identical={same(en)} code_identical={same([code_en])} rus={same(rus)} amh={same(amh)} hindi_identical={same(hin)} lossless={lossless}")
