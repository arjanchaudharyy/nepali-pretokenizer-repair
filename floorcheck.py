"""
floorcheck.py: is a tokenizer at its pre-token floor for a language, and would
the one-class regex repair help?

    python floorcheck.py <hf_repo_or_tokenizer.json> <flores_code | path/to/text.txt>

Examples:
    python floorcheck.py unsloth/Llama-3.2-1B npi_Deva
    python floorcheck.py Qwen/Qwen3-8B tam_Taml
    python floorcheck.py my_tokenizer/tokenizer.json my_corpus.txt

Prints:
  regex class        letters-only (\\p{L}+), mark-aware, or none
  tokens             tokens on the text
  floor              pre-tokens under the tokenizer's own regex: no vocabulary
                     extension that keeps this regex can use fewer tokens
  tokens / floor     close to 1.0 means vocabulary extension has no room left
  repaired floor     pre-tokens after admitting combining marks (\\p{M}) into the
                     word class; floor / repaired floor is the extra room the
                     repair gives
  premium            tokens(text) / tokens(English) on FLORES (parallel text only)

Only tokenizer files are downloaded; no model weights, no remote code.
"""
import json, sys, unicodedata
from pathlib import Path
import regex
from tokenizers import Tokenizer

FL = Path(__file__).parent / "flores200_dataset" / "devtest"


def load_tokenizer(arg):
    if arg.endswith(".json"):
        return Tokenizer.from_file(arg)
    from huggingface_hub import hf_hub_download
    return Tokenizer.from_file(hf_hub_download(arg, "tokenizer.json"))


def split_regexes(tk):
    pats = []

    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "Split" and "Regex" in n.get("pattern", {}):
                pats.append(n["pattern"]["Regex"])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(json.loads(tk.to_str()).get("pre_tokenizer"))
    return pats


def regex_class(pats):
    j = " ".join(pats)
    if not pats:
        return "none (no regex pre-tokenizer)"
    if "\\p{M" in j or "\\w" in j.replace("[^\\w", ""):
        return "mark-aware"
    if "\\p{L}" in j:
        return "letters-only (\\p{L}+)"
    return "other"


def repaired(p):
    return (p.replace(r"[^\r\n\p{L}\p{N}]?\p{L}+", r"[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+")
             .replace(r"[^\r\n\p{L}\p{N}]?+\p{L}++", r"[^\r\n\p{L}\p{M}\p{N}]?+[\p{L}\p{M}]++")
             .replace(r" ?\p{L}+", r" ?[\p{L}\p{M}]+")
             .replace(r" ?[^\s\p{L}\p{N}]+", r" ?[^\s\p{L}\p{M}\p{N}]+"))


def main(tok_arg, text_arg):
    tk = load_tokenizer(tok_arg)
    if (FL / f"{text_arg}.devtest").exists():
        lines = (FL / f"{text_arg}.devtest").read_text(encoding="utf-8").splitlines()
        en = (FL / "eng_Latn.devtest").read_text(encoding="utf-8").splitlines()
    else:
        lines, en = Path(text_arg).read_text(encoding="utf-8").splitlines(), None
    lines = [unicodedata.normalize("NFC", x) for x in lines if x.strip()]
    pats = split_regexes(tk)
    n_tok = sum(len(e.ids) for e in tk.encode_batch(lines, add_special_tokens=False))
    pre = tk.pre_tokenizer
    floor = sum(len(pre.pre_tokenize_str(x)) for x in lines) if pre else None
    rep_floor = None
    if pats:
        # count pre-tokens with the LAST (word-level) regex repaired; earlier splits (digits etc.) unchanged
        rx = [regex.compile(p) for p in pats[:-1]] + [regex.compile(repaired(pats[-1]))]

        def count(s):
            pieces = [s]
            for r in rx:
                pieces = [m for pc in pieces for m in (r.findall(pc) or [pc]) if m]
            return len(pieces)
        rep_floor = sum(count(x) for x in lines)
    print(f"tokenizer        {tok_arg}")
    print(f"regex class      {regex_class(pats)}")
    print(f"lines            {len(lines)}")
    print(f"tokens           {n_tok:,}")
    if floor:
        print(f"floor            {floor:,}   (no extension under this regex can go below)")
        print(f"tokens / floor   {n_tok / floor:.3f}   ({'little room left' if n_tok / floor < 1.15 else 'room left for extension'})")
    if rep_floor and floor:
        print(f"repaired floor   {rep_floor:,}   (extra room from the repair: {floor / rep_floor:.2f}x)")
    if en:
        e = sum(len(x.ids) for x in tk.encode_batch(en, add_special_tokens=False))
        print(f"premium          {n_tok / e:.2f}x English (FLORES-200 devtest)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
