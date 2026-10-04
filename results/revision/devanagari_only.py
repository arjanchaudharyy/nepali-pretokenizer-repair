"""
devanagari_only.py (review R1 W4 / W9): a Devanagari-scoped repair.

Variant regex: identical to the general repair (box/retrofit_tok.py `repaired`) except
that \\p{M} is replaced by an explicit class of the Devanagari-block combining marks
    U+0900-0903, U+093A-093C, U+093E-094F, U+0951-0957, U+0962-0963
(written as literal characters inside the classes). The class is verified against
every \\p{M} code point in U+0900-097F under (i) Python unicodedata, (ii) the `regex`
module, (iii) the HF `tokenizers` regex engine (Oniguruma) actually used at run time.
Because every character that is not one of these marks has the same class membership as
in the ORIGINAL regex, any text with no Devanagari mark pre-tokenizes exactly as under
the original regex (a strictly stronger guarantee than Proposition 1). Marks outside the
block (Vedic Extensions U+1CD0-1CFF, Devanagari Extended U+A8E0-A8FF) are NOT admitted.

(a) Nepali: pre-tokens under the Devanagari-only repair vs the general repair, for the
    Llama-3 and Qwen3 regexes, on FLORES devtest npi_Deva (1,012 sentences, NFC) and on all
    of data/ne_test_sample.txt (NFC lines, as box/retrofit_tok.py reads it). For the sample,
    identity is checked through the R2_K32000 tokenizer with each regex: equal token ids AND
    equal word_ids (pre-token index of every token) <=> identical pre-token segmentation.
    FLORES is additionally checked directly with pre_tokenize_str.
(b) All 204 FLORES devtest files: sentences whose pre-tokens differ from the ORIGINAL
    regex, under the general repair and under the Devanagari-only repair.

Run from the repo root:  .venv/bin/python results/revision/devanagari_only.py
Output: results/revision/devanagari_only.json
"""
import collections
import difflib
import json
import sys
import time
import unicodedata
from pathlib import Path
import regex
from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "box"))
from retrofit_tok import base_regex, repaired, set_regex  # noqa: E402

FL = ROOT / "flores200_dataset/devtest"
SAMPLE = ROOT / "data/ne_test_sample.txt"
BASES = {"Llama-3": ("unsloth/Llama-3.2-1B", "Llama-3.2-1B"), "Qwen3": ("Qwen/Qwen3-1.7B-Base", "Qwen3-1.7B-Base")}
RANGES = [(0x0900, 0x0903), (0x093A, 0x093C), (0x093E, 0x094F), (0x0951, 0x0957), (0x0962, 0x0963)]
DEVM = "".join(chr(a) if a == b else f"{chr(a)}-{chr(b)}" for a, b in RANGES)
DEVA_RX = regex.compile(r"[\u0900-\u097F]")
DEVM_SET = {c for a, b in RANGES for c in range(a, b + 1)}


def deva_only(pat_a):
    b = pat_a.replace(r"[^\r\n\p{L}\p{N}]?\p{L}+", rf"[^\r\n\p{{L}}{DEVM}\p{{N}}]?[\p{{L}}{DEVM}]+") \
             .replace(r" ?[^\s\p{L}\p{N}]+", rf" ?[^\s\p{{L}}{DEVM}\p{{N}}]+")
    assert b != pat_a
    return b


def with_regex(tk, pat):
    cfg = json.loads(tk.to_str())
    set_regex(cfg, pat)
    return Tokenizer.from_str(json.dumps(cfg))


def pre(tk, s):
    return [p for p, _ in tk.pre_tokenizer.pre_tokenize_str(s)]


sys.path.insert(0, str(ROOT))
from tokenizer_extras import U2B  # noqa: E402


def bl2s(p):  # byte-level pre-token -> readable text
    return bytes(U2B[c] for c in p).decode("utf-8", "replace")


def load(path):
    return [unicodedata.normalize("NFC", x) for x in path.read_text(encoding="utf-8").splitlines()]


def verify_class(base_tk):
    ud = {c for c in range(0x0900, 0x0980) if unicodedata.category(chr(c)).startswith("M")}
    rx = {c for c in range(0x0900, 0x0980) if regex.match(r"\p{M}", chr(c))}
    # tokenizers engine: a Split on \p{M} isolates the char iff it is a mark
    cfg = json.loads(base_tk.to_str())
    cfg["pre_tokenizer"] = {"type": "Split", "pattern": {"Regex": r"\p{M}"}, "behavior": "Isolated", "invert": False}
    t = Tokenizer.from_str(json.dumps(cfg))
    onig = set()
    for c in range(0x0900, 0x0980):
        s = "a" + chr(c) + "a"
        if [p for p, _ in t.pre_tokenizer.pre_tokenize_str(s)] == ["a", chr(c), "a"]:
            onig.add(c)
    # the Devanagari-only class as compiled by the tokenizers engine
    cfg["pre_tokenizer"]["pattern"]["Regex"] = f"[{DEVM}]"
    t2 = Tokenizer.from_str(json.dumps(cfg))
    cls = {c for c in range(0x0900, 0x0980)
           if [p for p, _ in t2.pre_tokenizer.pre_tokenize_str("a" + chr(c) + "a")] == ["a", chr(c), "a"]}
    hx = lambda S: [f"U+{c:04X}" for c in sorted(S)]
    return dict(unicodedata_version=unicodedata.unidata_version, n_marks=len(ud),
                unicodedata_eq_regex_module=ud == rx, unicodedata_eq_tokenizers=ud == onig,
                class_eq_unicodedata=DEVM_SET == ud, class_compiled_eq=cls == DEVM_SET,
                only_in_unicodedata=hx(ud - DEVM_SET), only_in_class=hx(DEVM_SET - ud),
                regex_module_diff=hx(ud ^ rx), tokenizers_diff=hx(ud ^ onig))


def main():
    t0 = time.time()
    out = dict(devm_ranges=[f"U+{a:04X}-U+{b:04X}" for a, b in RANGES], models={})
    langs = sorted(p.name.split(".")[0] for p in FL.glob("*.devtest"))
    flores = {c: load(FL / f"{c}.devtest") for c in langs}
    sample = [unicodedata.normalize("NFC", line) for line in open(SAMPLE, encoding="utf-8")]
    for name, (repo, short) in BASES.items():
        base = AutoTokenizer.from_pretrained(repo).backend_tokenizer
        pat_a = base_regex(json.loads(base.to_str()))
        pat_g, pat_d = repaired(pat_a), deva_only(pat_a)
        R = dict(original=pat_a, general=pat_g, deva_only=pat_d, class_check=verify_class(base))
        tO, tG, tD = base, with_regex(base, pat_g), with_regex(base, pat_d)
        # (a) FLORES Nepali, direct pre-tokens
        ne = flores["npi_Deva"]
        R["a_flores_npi_sentences_differ_G_vs_D"] = sum(pre(tG, s) != pre(tD, s) for s in ne)
        # (a) sample via R2_K32000 ids + word_ids, and FLORES the same way
        r2 = Tokenizer.from_file(str(ROOT / "tok/box" / short / "R2_K32000" / "tokenizer.json"))
        assert base_regex(json.loads(r2.to_str())) == pat_g
        r2d = with_regex(r2, pat_d)
        diff_lines, n_tok, CH = 0, 0, 2000
        diff_tokens, diff_lines_text = [], []
        for i in range(0, len(sample), CH):
            ch = sample[i:i + CH]
            eg = r2.encode_batch(ch, add_special_tokens=False)
            ed = r2d.encode_batch(ch, add_special_tokens=False)
            for a, b in zip(eg, ed):
                n_tok += len(a.ids)
                if a.ids != b.ids or a.word_ids != b.word_ids:
                    diff_lines += 1
                    diff_tokens.append((len(a.ids), len(b.ids)))
            ch_d = [s for s, a, b in zip(ch, eg, ed) if a.ids != b.ids or a.word_ids != b.word_ids]
            diff_lines_text.extend(ch_d)
        # diagnose the differing lines: which pre-tokens differ and which marks cause it
        marks, n_pg, n_diff_pg, n_diff_deva, n_seg, ex = collections.Counter(), 0, 0, 0, 0, []
        for s in diff_lines_text:
            pg, pd = pre(tG, s), pre(tD, s)
            n_pg += len(pg)
            sm = difflib.SequenceMatcher(a=pg, b=pd, autojunk=False)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "equal":
                    continue
                n_diff_pg += i2 - i1
                n_diff_deva += any(DEVA_RX.search(bl2s(x)) for x in pg[i1:i2] + pd[j1:j2])
                n_seg += 1
                if len(ex) < 8:
                    ex.append(dict(general=[bl2s(x) for x in pg[i1:i2]], deva_only=[bl2s(x) for x in pd[j1:j2]]))
            for chh in s:
                if unicodedata.category(chh).startswith("M") and ord(chh) not in DEVM_SET:
                    marks[f"U+{ord(chh):04X} {unicodedata.name(chh, '?')}"] += 1
        R["a_sample"] = dict(lines=len(sample), bytes=SAMPLE.stat().st_size, r2_tokens=n_tok,
                             lines_differ_G_vs_D=diff_lines,
                             r2_tokens_in_differing_lines_G_D=[sum(x for x, _ in diff_tokens), sum(y for _, y in diff_tokens)],
                             differing_pretokens_G=n_diff_pg, differing_segments=n_seg,
                             differing_segments_touching_devanagari=n_diff_deva, pretokens_in_differing_lines_G=n_pg,
                             non_devanagari_marks_in_differing_lines=dict(marks.most_common(15)),
                             examples=ex)
        R["a_flores_npi_R2_ids_differ_G_vs_D"] = sum(
            a.ids != b.ids for a, b in zip(r2.encode_batch(ne, add_special_tokens=False),
                                           r2d.encode_batch(ne, add_special_tokens=False)))
        print(name, "class", R["class_check"], "flores G!=D", R["a_flores_npi_sentences_differ_G_vs_D"],
              "sample", R["a_sample"], f"{time.time() - t0:.0f}s", flush=True)
        # (b) all FLORES languages
        per = {}
        for c in langs:
            g = d = gd = 0
            for s in flores[c]:
                po, pg, pd = pre(tO, s), pre(tG, s), pre(tD, s)
                g += po != pg
                d += po != pd
                gd += pg != pd
            per[c] = dict(general=g, deva_only=d, general_vs_deva_only=gd,
                          has_deva_mark=sum(any(ord(ch) in DEVM_SET for ch in s) for s in flores[c]))
        R["b_per_language"] = per
        R["b_langs_changed_general"] = sorted(c for c in per if per[c]["general"])
        R["b_langs_changed_deva_only"] = sorted(c for c in per if per[c]["deva_only"])
        # guarantee check: a sentence with no Devanagari mark never changes under deva_only
        R["b_violations_deva_only"] = 0
        for c in langs:
            for s in flores[c]:
                if not any(ord(ch) in DEVM_SET for ch in s) and pre(tO, s) != pre(tD, s):
                    R["b_violations_deva_only"] += 1
        out["models"][name] = R
        print(name, "general changes:", len(R["b_langs_changed_general"]), "langs;",
              "deva-only:", len(R["b_langs_changed_deva_only"]), R["b_langs_changed_deva_only"],
              "violations", R["b_violations_deva_only"], f"{time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(ROOT / "results/revision/devanagari_only.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
