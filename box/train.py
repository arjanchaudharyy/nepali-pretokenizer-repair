"""
train.py: continued pretraining (CPT), identical for every arm.

torchrun --nproc_per_node=4 train.py --model <dir> --data <dir> --out <dir>
         --lr 1e-4 --seed 0 [--max_steps N]

One epoch over packed 2048-token windows of the arm's token stream, in an order
fixed by --seed. Global batch = 256 windows (~0.5M tokens). AdamW (0.9, 0.95),
wd 0.1, grad clip 1.0, 1% linear warmup then cosine to 10% of peak. bf16
autocast with fp32 master weights. Because R2 encodes the same bytes in fewer
tokens it takes fewer steps; that is the point, and its FLOPs are logged.
"""
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--seq", type=int, default=2048)
p.add_argument("--global_batch", type=int, default=256)
p.add_argument("--micro", type=int, default=8)
p.add_argument("--max_steps", type=int, default=0)
p.add_argument("--frac", type=float, default=1.0)  # train on this PREFIX of the stream (same documents in every arm)
p.add_argument("--save", type=int, default=1)
p.add_argument("--ckpt", type=int, default=0)
p.add_argument("--save_at", type=int, default=0)  # also save a checkpoint after this many steps (equal-compute comparison)
a = p.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dev = torch.device("cuda", local)
torch.manual_seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True

tokens = np.memmap(Path(a.data) / "train.u32", dtype=np.uint32, mode="r")
W = a.seq + 1
n_win = int(((len(tokens) - 1) // a.seq) * a.frac)
order = np.random.default_rng(a.seed).permutation(n_win)
accum = a.global_batch // (a.micro * world)
assert accum * a.micro * world == a.global_batch
steps = n_win // a.global_batch
if a.max_steps:
    steps = min(steps, a.max_steps)
warm = max(1, int(0.01 * steps))

# Liger fused linear+cross-entropy: never materialises the (tokens x vocab) logits.
from liger_kernel.transformers import AutoLigerKernelForCausalLM
model = AutoLigerKernelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32,
                                                   attn_implementation="sdpa").to(dev)
model.config.use_cache = False
if a.ckpt:
    model.gradient_checkpointing_enable()
n_params = sum(p.numel() for p in model.parameters())
ddp = DDP(model, device_ids=[local])
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)


def lr_at(s):
    if s < warm:
        return a.lr * (s + 1) / warm
    t = (s - warm) / max(1, steps - warm)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def batch(step, micro_i):
    base = step * a.global_batch + (micro_i * world + rank) * a.micro
    idx = order[base:base + a.micro]
    x = np.stack([tokens[i * a.seq: i * a.seq + W] for i in idx]).astype(np.int64)
    return torch.from_numpy(x).to(dev, non_blocking=True)  # (micro, seq+1); HF shifts labels


out = Path(a.out)
if rank == 0:
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "w")
t0 = time.time()
for step in range(steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    tot = torch.zeros((), device=dev)
    for m in range(accum):
        x = batch(step, m)
        ctx = ddp.no_sync() if m < accum - 1 else torch.enable_grad()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ddp(input_ids=x, labels=x).loss
            (loss / accum).backward()
        tot += loss.detach() / accum
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if a.save_at and step + 1 == a.save_at and step + 1 < steps and rank == 0:
        import copy, shutil
        snap = copy.deepcopy(model).to(torch.bfloat16)
        snap.save_pretrained(out / f"model_step{a.save_at}", safe_serialization=True)
        for f in Path(a.model).iterdir():
            if f.name.startswith(("tokenizer", "special_tokens", "chat_template")):
                shutil.copy(f, out / f"model_step{a.save_at}" / f.name)
        del snap
    if step % 10 == 0 or step == steps - 1:
        dist.all_reduce(tot, op=dist.ReduceOp.AVG)
        if rank == 0:
            el = time.time() - t0
            rec = dict(step=step, steps=steps, loss=round(tot.item(), 4), lr=lr_at(step), gnorm=round(gn.item(), 3),
                       elapsed=round(el, 1), tok_per_s=round((step + 1) * a.global_batch * a.seq / el))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)

if rank == 0:
    n_tok = steps * a.global_batch * a.seq
    summary = dict(model=a.model, data=a.data, lr=a.lr, seed=a.seed, steps=steps, tokens=n_tok,
                   params=n_params, train_flops=6 * n_params * n_tok, secs=round(time.time() - t0, 1),
                   gpus=world, final_loss=tot.item())
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    if a.save:
        model.to(torch.bfloat16).save_pretrained(out / "model", safe_serialization=True)
        import shutil  # the checkpoint must carry its own tokenizer for evaluation
        for f in Path(a.model).iterdir():
            if f.name.startswith(("tokenizer", "special_tokens", "chat_template")):
                shutil.copy(f, out / "model" / f.name)
    print(summary, flush=True)
dist.barrier()
dist.destroy_process_group()
