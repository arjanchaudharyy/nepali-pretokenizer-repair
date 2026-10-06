"""Production audit: dedupe identical tokenizers, decompose premium into regex floor and residual."""
import json, unicodedata, regex
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/sweep_devtest.json"))
rows = [r for r in d["rows"]]
# group tokenizers whose token counts are identical on every language (same tokenizer under another name)
groups = {}
for r in rows:
    sig = tuple(r["premium"][c]["tokens"] for c in sorted(r["premium"]))
    groups.setdefault(sig, []).append(r)
uniq = []
for sig, rs in groups.items():
    rep = rs[0]
    rep = dict(rep, aliases=[x["name"] for x in rs[1:]])
    uniq.append(rep)
lossy = {r["name"] for r in uniq if not r["lossless"]["npi_Deva"]}
FL = Path("flores200_dataset/devtest")
ne = [unicodedata.normalize("NFC", x) for x in (FL / "npi_Deva.devtest").read_text().splitlines()]
out = []
for r in sorted(uniq, key=lambda r: r["premium"]["npi_Deva"]["premium"]):
    E = r["premium"]["eng_Latn"]["tokens"]
    floor = None
    if r["regexes"]:
        try:
            floor = sum(sum(len(regex.findall(p, s)) for p in r["regexes"][-1:]) for s in ne) / E
        except Exception:
            floor = None
    # Floors must count the FULL pre-tokenizer pipeline (Falcon3 also splits digit-like bytes after its
    # byte-level step); results/revision/full_pipeline_floor.py computes them for every HF tokenizer.
    FP = Path("results/revision/full_pipeline_floor.json")
    if FP.exists() and r["name"] in (fp := json.load(open(FP))):
        floor = fp[r["name"]]["full_pipeline_floor"]
    p = r["premium"]["npi_Deva"]
    out.append(dict(name=r["name"], aliases=r["aliases"], family=r["family"], vocab=r["vocab"], kind=r["kind"],
                    premium=p["premium"], lo=p["lo"], hi=p["hi"], lossless=r["name"] not in lossy,
                    floor=floor, mark_initial=r["integrity"]["npi_Deva"]["mark_initial"],
                    break_rate=r["integrity"]["npi_Deva"]["break_rate"],
                    intra_cp=r["integrity"]["npi_Deva"]["intra_cp"]))
json.dump(out, open("results/audit_unique.json", "w"), indent=1, ensure_ascii=False)
for o in out:
    print(f"{o['name']:<13}{o['kind']:<13}{o['vocab']:>8} {o['premium']:.2f} [{o['lo']:.2f},{o['hi']:.2f}] floor={o['floor'] if o['floor'] is None else round(o['floor'],2)} lossless={o['lossless']} aliases={o['aliases']}")
