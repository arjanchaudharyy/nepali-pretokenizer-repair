"""
init_model.py: resize a base model to an extended tokenizer and initialise
the new rows.

Each new token's string is tokenized by the BASE tokenizer; its new input-
embedding row (and output row, if untied) is the mean of those base rows
(fast vocabulary transfer, Gee et al. 2022). Old rows are untouched, so on any
text that tokenizes to old ids only (all English, by construction), the
initialised model computes exactly what the base model computes. We verify
that numerically before saving.

usage: python init_model.py <base_repo> <ext_tokenizer_dir_or_none> <out_dir>
"""
import json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


def main(base_repo, ext_dir, out_dir):
    out = Path(out_dir)
    model = AutoModelForCausalLM.from_pretrained(base_repo, torch_dtype=torch.float32)
    base_tok = AutoTokenizer.from_pretrained(base_repo)
    if ext_dir == "none":
        model.save_pretrained(out, safe_serialization=True)
        base_tok.save_pretrained(out)
        print("saved base unchanged")
        return
    ext_tok = PreTrainedTokenizerFast.from_pretrained(f"{ext_dir}/hf")
    meta = json.load(open(f"{ext_dir}/meta.json"))
    first = meta["first_new_id"]
    n_new = meta["new_tokens"]
    old_in = model.get_input_embeddings().weight.data.clone()
    tied = model.config.tie_word_embeddings
    old_out = None if tied else model.get_output_embeddings().weight.data.clone()
    model.resize_token_embeddings(first + n_new, mean_resizing=False)
    emb_in = model.get_input_embeddings().weight.data
    emb_out = None if tied else model.get_output_embeddings().weight.data
    bt = base_tok.backend_tokenizer
    # rows between the old matrix size and `first` (padding rows some models have)
    # keep their resized values; we only initialise the new token rows.
    for k in range(n_new):
        i = first + k
        # Constituents come from the BASE merges applied to the token's own byte-level
        # string (no pre-tokenizer, no decoding), so tokens that end inside a UTF-8
        # character are handled exactly instead of decoding to U+FFFD.
        ids = [t.id for t in bt.model.tokenize(ext_tok.convert_ids_to_tokens(i))]
        assert ids and max(ids) < first, (i, ids)
        emb_in[i] = old_in[ids].mean(0)
        if emb_out is not None:
            emb_out[i] = old_out[ids].mean(0)
    # numerical check: English logits on old ids are unchanged
    model.eval()
    probe = "The government will begin reconstruction soon, officials said on Tuesday."
    ids = torch.tensor([bt.encode(probe, add_special_tokens=False).ids])
    assert ext_tok(probe, add_special_tokens=False)["input_ids"] == ids[0].tolist()
    base = AutoModelForCausalLM.from_pretrained(base_repo, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        # compare only the real original vocabulary (Qwen pads its matrix past `first`,
        # and those padding rows are reused for new tokens)
        a = base(ids).logits[0, :, :first]
        b = model(ids).logits[0, :, :first]
    diff = (a - b).abs().max().item()
    assert diff < 1e-4, diff
    model.save_pretrained(out, safe_serialization=True)
    ext_tok.save_pretrained(out)
    print(json.dumps(dict(base=base_repo, ext=ext_dir, new_rows=n_new, first_new_id=first,
                          tied=tied, english_logit_maxdiff=diff)))


if __name__ == "__main__":
    main(*sys.argv[1:4])
