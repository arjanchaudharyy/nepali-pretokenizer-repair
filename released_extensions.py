"""
released_extensions.py: measure RELEASED vocabulary extensions of letters-only
byte-level BPE tokenizers (Llama-3/3.1/3.2, Qwen2.5/3, LFM2) against the
pre-token floor of their own pre-tokenizer.

For a tokenizer whose model is BPE over the pre-tokens of a regex Split, every
pre-token becomes >= 1 token, so  tokens >= pre-tokens  whatever merges are
added. HF `added_tokens` are matched BEFORE the pre-tokenizer runs and can
therefore cross pre-token boundaries ("bypass"); a changed regex moves the floor.

Per (repo, FLORES language), on FLORES-200 devtest (one line per sentence, NFC):
  base_tokens       tokens under the base model's released tokenizer
  ext_tokens        tokens under the released extension
  pretokens         pre-tokens under the extension's own pre-tokenizer
                    (normalizer, if any, then backend pre_tokenize_str)
  ext_over_floor    ext_tokens / pretokens       (1.0 = sits exactly on the floor)
  pretokens_repaired  pre-tokens under the mark-aware repair of the extension's
                    regex (box/retrofit_tok.py: repaired()); only for letters-only regexes
  repair_headroom   pretokens / pretokens_repaired (how far the repair lowers the floor)
  premium_ext / premium_base   target tokens / eng_Latn tokens with the same tokenizer

Only tokenizer.json, tokenizer_config.json and special_tokens_map.json are
downloaded (never weights); nothing from a model repo is executed
(no trust_remote_code anywhere).

usage: python released_extensions.py            -> results/released_extensions.json
"""
from __future__ import annotations
import json, sys, time, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "box"))
from retrofit_tok import repaired, set_regex  # noqa: E402  (the repair used in the paper)

from huggingface_hub import snapshot_download  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

FLORES = ROOT / "flores200_dataset" / "devtest"
OUT = ROOT / "results" / "released_extensions.json"
TOK_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

# Base tokenizers (gated first-party repos replaced by byte-identical mirrors, as in tok_registry.py)
BASES = {
    "Llama-3-8B": "NousResearch/Meta-Llama-3-8B",
    "Llama-3.1-8B": "unsloth/Meta-Llama-3.1-8B",
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Qwen3-14B-Base": "Qwen/Qwen3-14B-Base",
    "LFM2-8B-A1B": "LiquidAI/LFM2-8B-A1B",
}

# (repo, base key, FLORES codes, source). Hardcoded so the table is reproducible.
REPOS = [
    # Yamaguchi et al. (Computational Linguistics 2026; ElChat, TMLR 2025): continued-merge VE + mean init
    ("atsuki-yamaguchi/Llama-3-8B-te-30K-5000-mean", "Llama-3-8B", ["tel_Telu"], "Yamaguchi et al. 2026"),
    ("atsuki-yamaguchi/Llama-3-8B-si-30K-5000-mean", "Llama-3-8B", ["sin_Sinh"], "Yamaguchi et al. 2026"),
    ("atsuki-yamaguchi/Llama-3-8B-my-30K-5000-mean", "Llama-3-8B", ["mya_Mymr"], "Yamaguchi et al. 2026"),
    ("atsuki-yamaguchi/Llama-3.1-8B-ta-madlad-mean-tuned", "Llama-3.1-8B", ["tam_Taml"], "Yamaguchi et al. 2025 (ElChat)"),
    ("atsuki-yamaguchi/Llama-3.1-8B-bn-madlad-mean-tuned", "Llama-3.1-8B", ["ben_Beng"], "Yamaguchi et al. 2025 (ElChat)"),
    ("atsuki-yamaguchi/Llama-3.1-8B-gu-madlad-mean-tuned", "Llama-3.1-8B", ["guj_Gujr"], "Yamaguchi et al. 2025 (ElChat)"),
    ("atsuki-yamaguchi/Llama-3.1-8B-am-madlad-mean-tuned", "Llama-3.1-8B", ["amh_Ethi"], "Yamaguchi et al. 2025 (ElChat); control: Ethiopic has no combining marks"),
    ("atsuki-yamaguchi/Qwen2.5-7B-si-madlad-mean-tuned", "Qwen2.5-7B", ["sin_Sinh"], "Yamaguchi et al. 2025 (ElChat)"),
    ("atsuki-yamaguchi/Qwen2.5-7B-my-madlad-mean-tuned", "Qwen2.5-7B", ["mya_Mymr"], "Yamaguchi et al. 2025 (ElChat)"),
    ("atsuki-yamaguchi/Qwen3-14B-Base-bn-madlad-mean-tuned", "Qwen3-14B-Base", ["ben_Beng"], "Yamaguchi et al. (Qwen3 release)"),
    ("atsuki-yamaguchi/Qwen3-14B-Base-te-madlad-mean-tuned", "Qwen3-14B-Base", ["tel_Telu"], "Yamaguchi et al. (Qwen3 release)"),
    # Smith et al. 2026 (Liquid AI), In-Place Tokenizer Expansion: LFM2 64K -> LFM2.5 ~125K
    ("LiquidAI/LFM2.5-8B-A1B", "LFM2-8B-A1B", ["hin_Deva", "ben_Beng", "npi_Deva", "tha_Thai"], "Smith et al. 2026"),
    # Regex changed (TituLLMs, Nahin et al. 2025)
    ("hishab/titulm-llama-3.2-1b-v2.0", "Llama-3.2-1B", ["ben_Beng"], "Nahin et al. 2025 (TituLLMs)"),
    # added_tokens instead of merges
    ("polyglots/SinLlama_v01", "Llama-3-8B", ["sin_Sinh"], "Aravinda et al. 2025 (SinLlama)"),
    ("Anurag-Tiwari/hindi_updated-llama-tokenizer", "Llama-3-8B", ["hin_Deva"], "grey literature (HF only)"),
    # Gated: recorded as inaccessible if no token grants access
    ("MBZUAI/Llama-3-Nanda-10B-Chat", "Llama-3-8B", ["hin_Deva"], "Singh et al. 2026 (Nanda)"),
]


def fetch(repo):
    d = snapshot_download(repo, allow_patterns=TOK_FILES)
    p = Path(d) / "tokenizer.json"
    if not p.exists():
        raise FileNotFoundError(f"{repo}: no tokenizer.json")
    return p


def split_regexes(cfg):
    out = []

    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "Split" and "Regex" in n.get("pattern", {}):
                out.append(n["pattern"]["Regex"])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(cfg.get("pre_tokenizer"))
    return out


def letters_only(pat):
    return r"\p{L}+" in pat and r"\p{M}" not in pat and r"\w" not in pat


def lines(code):
    p = FLORES / f"{code}.devtest"
    return [unicodedata.normalize("NFC", s) for s in p.read_text(encoding="utf-8").splitlines()]


def n_tokens(tk, sents):
    return sum(len(e.ids) for e in tk.encode_batch(sents, add_special_tokens=False))


def n_pretokens(tk, sents):
    """Pre-tokens of the tokenizer's own pipeline: normalizer then pre_tokenizer.
    Added tokens are deliberately NOT extracted, so the count is the regex floor."""
    norm, pre = tk.normalizer, tk.pre_tokenizer
    n = 0
    for s in sents:
        if norm is not None:
            s = norm.normalize_str(s)
        n += sum(1 for piece, _ in pre.pre_tokenize_str(s) if piece)
    return n


def describe(base_cfg, ext_cfg):
    bv, ev = base_cfg["model"]["vocab"], ext_cfg["model"]["vocab"]
    norm = lambda m: [" ".join(x) if isinstance(x, list) else x for x in m]  # noqa: E731
    bm, em = norm(base_cfg["model"]["merges"]), norm(ext_cfg["model"]["merges"])
    b_added = {t["content"] for t in base_cfg.get("added_tokens", [])}
    e_added = [t for t in ext_cfg.get("added_tokens", []) if t["content"] not in b_added]
    sm = set(em)
    return dict(
        base_model_vocab=len(bv), ext_model_vocab=len(ev),
        base_merges=len(bm), ext_merges=len(em),
        base_strings_kept=sum(k in ev for k in bv),
        base_ids_kept=sum(ev.get(k) == v for k, v in bv.items()),
        base_merges_kept=sum(m in sm for m in bm),
        new_vocab_entries=sum(k not in bv for k in ev),
        new_added_tokens=len(e_added),
        new_added_nonspecial=sum(1 for t in e_added if not t.get("special")),
    )


def main():
    t0 = time.time()
    eng = lines("eng_Latn")
    cache, rows, inaccessible = {}, [], []

    def get(repo):
        if repo not in cache:
            p = fetch(repo)
            cache[repo] = (Tokenizer.from_file(str(p)), json.loads(p.read_text(encoding="utf-8")))
        return cache[repo]

    for repo, base_key, codes, source in REPOS:
        base_repo = BASES[base_key]
        try:
            ext, ext_cfg = get(repo)
        except Exception as ex:
            inaccessible.append(dict(repo=repo, base=base_repo, langs=codes, source=source,
                                     error=f"{type(ex).__name__}: {str(ex).splitlines()[0][:200]}"))
            print(f"SKIP {repo}: {type(ex).__name__}", flush=True)
            continue
        base, base_cfg = get(base_repo)
        rx_b, rx_e = split_regexes(base_cfg), split_regexes(ext_cfg)
        info = describe(base_cfg, ext_cfg)
        regex_changed = rx_b != rx_e
        lo = all(letters_only(p) for p in rx_e) and len(rx_e) == 1
        if info["new_added_nonspecial"] > 1000:
            mech = "added_tokens"
        elif info["new_vocab_entries"] > 0:
            mech = "merges"
        else:
            mech = "none"
        rep_tk = None
        if lo:
            rep_cfg = json.loads(json.dumps(ext_cfg))
            set_regex(rep_cfg, repaired(rx_e[0]))
            rep_tk = Tokenizer.from_str(json.dumps(rep_cfg))
        base_tk_noadd = None
        if regex_changed:  # floor under the BASE regex, for comparison
            base_tk_noadd = base
        eng_b, eng_e = n_tokens(base, eng), n_tokens(ext, eng)
        for code in codes:
            s = lines(code)
            bt, et = n_tokens(base, s), n_tokens(ext, s)
            pt = n_pretokens(ext, s)
            row = dict(
                repo=repo, base=base_repo, base_key=base_key, lang=code, source=source,
                base_vocab=base.get_vocab_size(with_added_tokens=True),
                ext_vocab=ext.get_vocab_size(with_added_tokens=True),
                mechanism=mech, regex_changed=regex_changed, ext_regex_letters_only=lo,
                ext_regex=rx_e, base_regex=rx_b, **info,
                sentences=len(s), bytes=sum(len(x.encode()) for x in s),
                base_tokens=bt, ext_tokens=et, pretokens=pt,
                ext_over_floor=et / pt, base_over_floor=bt / pt,
                compression_vs_base=bt / et,
                eng_tokens_base=eng_b, eng_tokens_ext=eng_e,
                premium_base=bt / eng_b, premium_ext=et / eng_e,
            )
            if rep_tk is not None:
                pr = n_pretokens(rep_tk, s)
                row.update(pretokens_repaired=pr, repair_headroom=pt / pr, ext_over_repaired_floor=et / pr)
            if base_tk_noadd is not None:
                pb = n_pretokens(base_tk_noadd, s)
                row.update(pretokens_base_regex=pb, ext_over_base_floor=et / pb)
                if all(letters_only(p) for p in rx_b) and len(rx_b) == 1:
                    rb = json.loads(json.dumps(base_cfg))
                    set_regex(rb, repaired(rx_b[0]))
                    row.update(pretokens_base_repaired=n_pretokens(Tokenizer.from_str(json.dumps(rb)), s))
            rows.append(row)
            print(f"{repo:52s} {code} base={bt:,} ext={et:,} floor={pt:,} ext/floor={et/pt:.3f}"
                  + (f" repaired={row['pretokens_repaired']:,} headroom={row['repair_headroom']:.2f}x"
                     if "pretokens_repaired" in row else "") + f" ({time.time()-t0:.0f}s)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(split="devtest", note=__doc__.strip().split("\n\n")[0], rows=rows, inaccessible=inaccessible),
              open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {OUT} ({len(rows)} rows, {len(inaccessible)} inaccessible)")


if __name__ == "__main__":
    main()
