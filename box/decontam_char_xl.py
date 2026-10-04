"""Supplementary char-level decontamination for scripts without word spaces
(Thai, Burmese), where whitespace 10-grams cover few FLORES sentences.
Drops docs sharing any >=74-char span with FLORES dev/devtest (char 50-grams of
FLORES at every offset; doc windows at stride 25). Rewrites data/xl/<code>.txt.
usage: python box/decontam_char_xl.py <code>"""
import json, sys, unicodedata
from pathlib import Path
H = Path(__file__).resolve().parent.parent
N, S = 50, 25
code = sys.argv[1]
g = set()
for sp in ("dev", "devtest"):
    for l in (H / "flores200_dataset" / sp / f"{code}.{sp}").read_text(encoding="utf-8").splitlines():
        l = unicodedata.normalize("NFC", l)
        for i in range(len(l) - N + 1):
            g.add(hash(l[i:i + N]))
p = H / "data/xl" / f"{code}.txt"
lines = p.read_text(encoding="utf-8").splitlines()
keep, drop = [], 0
for t in lines:
    if any(hash(t[i:i + N]) in g for i in range(0, len(t) - N + 1, S)):
        drop += 1
    else:
        keep.append(t)
p.write_text("".join(x + "\n" for x in keep), encoding="utf-8")
st = json.load(open(H / "data/xl" / f"prep.{code}.json"))
st["char50_contam"] = drop
st["kept_lines"] = len(keep)  # file is line-based; doc count unchanged unless drop>0
st["bytes"] = sum(len(x.encode()) + 1 for x in keep)
json.dump(st, open(H / "data/xl" / f"prep.{code}.json", "w"), indent=1)
print(code, "char-contaminated docs dropped:", drop, "kept", len(keep), st["bytes"])
