# BOS vs end-of-text prefix diagnostic (output recorded in results/bos_check.txt)
import torch, json, math
from transformers import AutoModelForCausalLM, AutoTokenizer
docs=[json.loads(l) for _,l in zip(range(40), open("corpus/npi_Deva.heldout.jsonl"))]
docs=[d[:1500] for d in docs]
for d in ["models/llama_R0","runs/llama_R0_s0/model","runs/B1_llama_R0/model"]:
    tk=AutoTokenizer.from_pretrained(d); m=AutoModelForCausalLM.from_pretrained(d,dtype=torch.bfloat16).cuda().eval()
    res={}
    for name,pre in [("bos",[tk.bos_token_id]),("eos",[tk.eos_token_id])]:
        nll=0;nb=0
        for t in docs:
            ids=pre+tk(t,add_special_tokens=False)["input_ids"]
            x=torch.tensor([ids]).cuda()
            with torch.no_grad(): lg=m(input_ids=x).logits[0].float()
            nll+=torch.nn.functional.cross_entropy(lg[:-1],x[0,1:],reduction="sum").item(); nb+=len(t.encode())
        res[name]=round(nll/math.log(2)/nb,4)
    e=m.get_input_embeddings().weight.float()
    print(d,res,"bos_norm",round(e[tk.bos_token_id].norm().item(),3),"eos_norm",round(e[tk.eos_token_id].norm().item(),3),"mean_norm",round(e.norm(dim=1).mean().item(),3))
    del m; torch.cuda.empty_cache()
