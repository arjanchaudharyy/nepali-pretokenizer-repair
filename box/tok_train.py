"""
tok_train.py: train one byte-level BPE tokenizer for a single cell of the grid.

The ONLY thing that differs between arms is the pre-tokenization regex:
  A (letter): Llama-3 / cl100k regex, verbatim from Meta-Llama-3 tokenizer.json
  B (mark)  : A with \\p{M} admitted wherever \\p{L} is a word class, and excluded
              from the leading-character and punctuation classes.
Everything else is shared: NFC text, byte-level alphabet, BpeTrainer settings,
corpus bytes (the identical sampled file is used by both arms), special tokens.

usage: python tok_train.py <arm A|B> <vocab> <lang_code> <lang_share> <total_bytes> <out.json>
"""
import random, sys, time, json
from pathlib import Path
from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders, trainers

H = Path("/home/ntt")

PAT_A = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
PAT_B = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
PATS = {"A": PAT_A, "B": PAT_B}
SPECIAL = ["<|endoftext|>"]


def build(arm: str) -> Tokenizer:
    tk = Tokenizer(models.BPE(byte_fallback=False))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATS[arm]), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    return tk


def sample_file(code: str, share: float, total: int) -> Path:
    """Deterministic mixture file, shared by both arms (cached by name)."""
    p = H / "tok" / f"mix_{code}_{share:g}_{total}.txt"
    if p.exists():
        return p
    rng = random.Random(0)
    tmp = p.with_suffix(".part")
    with open(tmp, "w", encoding="utf-8") as out:
        for c, budget in ((code, int(total * share)), ("eng_Latn", int(total * (1 - share)))):
            got = 0
            with open(H / "corpus" / f"{c}.txt", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    got += len(line.encode())
                    if got >= budget:
                        break
    tmp.rename(p)
    return p


def main(arm, vocab, code, share, total, out):
    t0 = time.time()
    mix = sample_file(code, float(share), int(float(total)))
    tk = build(arm)
    tr = trainers.BpeTrainer(vocab_size=int(vocab), min_frequency=2, special_tokens=SPECIAL,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)

    def lines():
        with open(mix, encoding="utf-8") as f:
            for line in f:
                yield line

    tk.train_from_iterator(lines(), trainer=tr)
    # lossless check on a mixed probe; refuse to save a lossy artifact
    probe = (H / "flores200_dataset/dev/npi_Deva.dev").read_text(encoding="utf-8")[:20000] + \
            (H / "flores200_dataset/dev/eng_Latn.dev").read_text(encoding="utf-8")[:20000]
    assert tk.decode(tk.encode(probe).ids) == probe, "lossy tokenizer"
    tk.save(out)
    print(json.dumps(dict(arm=arm, vocab=int(vocab), code=code, share=float(share), total=int(float(total)),
                          out=out, secs=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:7])
