"""
tok_registry.py: one loader for every tokenizer in the study.

Each entry yields (name, family, vocab, encode, decode, pretok_regex | None, kind)
where kind is one of
    "letter-regex"  regex pre-tokenizer whose word class is \\p{L}+ (marks excluded)
    "mark-regex"    regex pre-tokenizer whose word class admits \\p{M}
    "no-regex"      SentencePiece / whitespace-only pre-tokenization
The kind is DERIVED from the tokenizer's own config, never hand-labelled.
"""
from __future__ import annotations
import json, os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# (label, family, repo). Gated first-party repos are replaced by byte-identical mirrors.
HF = [
    ("Llama-3", "Meta", "NousResearch/Meta-Llama-3-8B"),
    ("Llama-4", "Meta", "unsloth/Llama-4-Scout-17B-16E-Instruct"),
    ("Gemma-2", "Google", "unsloth/gemma-2-9b"),
    ("Gemma-3", "Google", "unsloth/gemma-3-12b-it"),
    ("Qwen2.5", "Alibaba", "Qwen/Qwen2.5-7B"),
    ("Qwen3", "Alibaba", "Qwen/Qwen3-8B"),
    ("DeepSeek-V3", "DeepSeek", "deepseek-ai/DeepSeek-V3"),
    ("Mistral-NeMo", "Mistral", "unsloth/Mistral-Nemo-Base-2407"),
    ("Mistral-7B", "Mistral", "unsloth/mistral-7b-v0.3"),
    ("Phi-4", "Microsoft", "microsoft/phi-4"),
    ("GLM-4.5", "Zhipu", "zai-org/GLM-4.5"),
    ("gpt-oss", "OpenAI", "openai/gpt-oss-20b"),
    ("OLMo-2", "AI2", "allenai/OLMo-2-1124-7B"),
    ("SmolLM3", "HF", "HuggingFaceTB/SmolLM3-3B"),
    ("Falcon3", "TII", "tiiuae/Falcon3-7B-Base"),
    ("BLOOM", "BigScience", "bigscience/bloom"),
    ("mT5", "Google", "google/mt5-base"),
    ("XLM-R", "Meta", "FacebookAI/xlm-roberta-base"),
    ("NLLB-200", "Meta", "facebook/nllb-200-distilled-600M"),
    ("IndicBERTv2", "AI4Bharat", "ai4bharat/IndicBERTv2-MLM-only"),
    ("Sarvam-1", "Sarvam", "sarvamai/sarvam-1"),
    ("Sarvam-M", "Sarvam", "sarvamai/sarvam-m"),
    ("Arkios", "Arkios", "sajalregmi4/arkios-tokenizer"),
    # 2026 releases
    ("Qwen3.5", "Alibaba", "Qwen/Qwen3.5-9B"),
    ("Kimi-K3", "Moonshot", "moonshotai/Kimi-K3"),
    ("Gemma-4", "Google", "google/gemma-4-12B-it"),
    ("DeepSeek-V4", "DeepSeek", "deepseek-ai/DeepSeek-V4-Flash"),
    ("GLM-5.3", "Zhipu", "zai-org/GLM-5.3"),
    ("Mistral-Large-3", "Mistral", "mistralai/Mistral-Large-3-675B-Instruct-2512"),
    ("Granite-4.1", "IBM", "ibm-granite/granite-4.1-3b"),
    # published vocabulary-extension retrofits of letters-only models
    ("LFM2", "Liquid", "LiquidAI/LFM2-8B-A1B"),
    ("TituLLM", "Hishab", "hishab/titulm-llama-3.2-1b-v2.0"),
]
TIKTOKEN = [("GPT-2", "OpenAI", "gpt2"), ("cl100k", "OpenAI", "cl100k_base"),
            ("o200k", "OpenAI", "o200k_base")]


def classify(regexes: list[str]) -> str:
    if not regexes:
        return "no-regex"
    joined = " ".join(regexes)
    # mark-aware if any word class includes \p{M} (or the \p{Mn}/\p{Mc} subclasses)
    if "\\p{M" in joined or "\\p{Mn" in joined:
        return "mark-regex"
    # Oniguruma's \w includes combining marks (Mn/Mc), so a \w-based word
    # class keeps vowel signs attached (verified on TituLLM's released regex).
    if "\\w" in joined.replace("[^\\w", ""):
        return "mark-regex"
    if "\\p{L}" in joined or "\\p{Lu}" in joined or "\\p{Lo}" in joined:
        return "letter-regex"
    return "other-regex"


def _hf_regexes(tk) -> list[str]:
    try:
        cfg = json.loads(tk.backend_tokenizer.to_str())
    except Exception:
        return []
    out = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "Split":
                pat = node.get("pattern", {})
                if "Regex" in pat:
                    out.append(pat["Regex"])
            if node.get("type") == "ByteLevel" and node.get("use_regex", False):
                # HF's ByteLevel default regex is GPT-2's: \p{L}+ word class
                out.append("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(cfg.get("pre_tokenizer"))
    return out


def load(names=None):
    out, skipped = [], []
    import tiktoken
    for label, fam, enc_name in TIKTOKEN:
        if names and label not in names:
            continue
        e = tiktoken.get_encoding(enc_name)
        out.append(dict(name=label, family=fam, vocab=e.n_vocab,
                        encode=lambda t, e=e: e.encode(t, disallowed_special=()),
                        decode=lambda i, e=e: e.decode(i),
                        decode_bytes=lambda i, e=e: e.decode_single_token_bytes(i),
                        regexes=[e._pat_str], kind=classify([e._pat_str])))
    from transformers import AutoTokenizer
    for label, fam, repo in HF:
        if names and label not in names:
            continue
        try:
            tk = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
            rx = _hf_regexes(tk)
            out.append(dict(name=label, family=fam, vocab=len(tk),
                            encode=lambda t, tk=tk: tk.encode(t, add_special_tokens=False),
                            decode=lambda i, tk=tk: tk.decode(i, skip_special_tokens=True,
                                                              clean_up_tokenization_spaces=False),
                            decode_bytes=None, regexes=rx, kind=classify(rx), repo=repo))
        except Exception as ex:
            skipped.append((label, f"{type(ex).__name__}: {str(ex)[:160]}"))
    return out, skipped


def load_file(path: str, name: str, family: str = "this work"):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(path))
    cfg = json.loads(tk.to_str())
    rx = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "Split" and "Regex" in node.get("pattern", {}):
                rx.append(node["pattern"]["Regex"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(cfg.get("pre_tokenizer"))
    return dict(name=name, family=family, vocab=tk.get_vocab_size(),
                encode=lambda t: tk.encode(t, add_special_tokens=False).ids,
                decode=lambda i: tk.decode(i, skip_special_tokens=False),
                decode_bytes=None, regexes=rx, kind=classify(rx))
