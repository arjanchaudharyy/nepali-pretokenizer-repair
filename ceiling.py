"""Pre-token ceilings and the English-lossless property of the regex repair."""
import json, unicodedata, regex
from pathlib import Path
from transformers import AutoTokenizer
FL=Path("flores200_dataset")
PAT_A = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
PAT_B = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
A,Bp=regex.compile(PAT_A),regex.compile(PAT_B)
def load(c,s="devtest"): return [unicodedata.normalize("NFC",x) for x in (FL/s/f"{c}.{s}").read_text().splitlines()]
out={}
en=load("eng_Latn"); ne=load("npi_Deva")
# 1) pre-token counts (the floor any BPE vocabulary can reach, ignoring the byte level)
for c in ["npi_Deva","hin_Deva","eng_Latn","amh_Ethi","rus_Cyrl"]:
    t=load(c); a=sum(len(A.findall(s)) for s in t); b=sum(len(Bp.findall(s)) for s in t)
    out[c]=dict(pretok_A=a,pretok_B=b,ratio=a/b); print(c,a,b,round(a/b,3))
# 2) English identity: do A and B produce identical pre-token sequences on English text?
diff=sum(A.findall(s)!=Bp.findall(s) for s in en); print("FLORES eng sentences whose pre-tokens differ:",diff,"/",len(en))
out["eng_pretok_diff_sentences"]=diff
json.dump(out,open("results/ceiling.json","w"),indent=1)
