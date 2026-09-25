"""One-shot CDQ-LRR on ANY HF model (built for MiMo-V2.6-Distill-Qwen-9B).

Install -> download -> quantize -> validate -> report. No GPU needed for the
quant pass (CPU, per-tensor streaming; ~2x model size RAM peak).

    python code/run_any_model.py --model-id XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B

Needs: torch, transformers (very new for qwen3_5 arch), safetensors,
huggingface_hub. See requirements.txt.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_CODE_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, _CODE_DIR)

DEFAULT_SUBS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args():
    ap = argparse.ArgumentParser(description="CDQ-LRR one-shot runner for any HF model")
    ap.add_argument("--model-id", default="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B")
    ap.add_argument("--local-dir", default=None, help="use local checkpoint dir, skip download")
    ap.add_argument("--lattice", type=int, choices=(26, 16), default=26)
    ap.add_argument("--stc-frac", type=float, default=0.0156)
    ap.add_argument("--targets", default=DEFAULT_SUBS,
                    help="comma-separated Linear name substrings to quantize")
    ap.add_argument("--prompt", default="Explain the mechanism of action of beta blockers in 2 sentences.")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--report", default="cdq_trial_report.json")
    ap.add_argument("--save-dequant", default=None, help="optional dir to save CDQ-dequantized checkpoint")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    return ap.parse_args()


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    args = parse_args()
    from cdq_lrr_qwen05 import CAMKII_LATTICE, CAMKII_LATTICE_16, quantize_cdq_lrr, dequantize

    lattice = CAMKII_LATTICE if args.lattice == 26 else CAMKII_LATTICE_16
    subs = tuple(s.strip() for s in args.targets.split(",") if s.strip())

    # 1. download (skipped if local dir given)
    if args.local_dir:
        src = args.local_dir
    else:
        from huggingface_hub import snapshot_download
        print(f"[1/5] downloading {args.model_id} ...", flush=True)
        src = snapshot_download(repo_id=args.model_id,
                                allow_patterns=["*.safetensors", "*.json", "tokenizer*",
                                                "merges.txt", "vocab.*", "*.jinja"])
        print(f"  -> {src}", flush=True)

    # 2. load baseline
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[2/5] loading baseline ...", flush=True)
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=args.trust_remote_code, low_cpu_mem_usage=True).eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {total_params/1e9:.2f}B", flush=True)

    # baseline logits BEFORE patching (peak RAM = 1 model, not 2)
    enc = tok(args.prompt, return_tensors="pt")
    with torch.no_grad():
        base_logits = model(**enc).logits.float()

    # 3. quantize in place
    print(f"[3/5] CDQ-LRR lattice={args.lattice} stc={args.stc_frac} targets={subs}", flush=True)
    t0 = time.time()
    packs, mses, coss, tot_w, tot_res = {}, [], [], 0, 0
    n_mod = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and mod.weight.dim() == 2 \
                    and any(s in name for s in subs):
                ow = mod.weight.data.to(torch.bfloat16)
                pack = quantize_cdq_lrr(ow, lattice=lattice, stc_frac=args.stc_frac)
                rw = dequantize(pack).to(torch.bfloat16)
                mses.append((ow.float() - rw.float()).pow(2).mean().item())
                coss.append(F.cosine_similarity(ow.flatten().float(),
                                                rw.flatten().float(), dim=0).item())
                mod.weight.data = rw
                tot_w += pack["numel"]
                tot_res += pack["outlier_count"]
                n_mod += 1
    dt = time.time() - t0
    avg_mse = sum(mses) / len(mses)
    avg_cos = sum(coss) / len(coss)
    print(f"  {n_mod} linears, {tot_w/1e6:.1f}M params, {tot_res} residuals "
          f"({100*tot_res/max(1,tot_w):.2f}%) in {dt:.0f}s", flush=True)
    print(f"  weight MSE={avg_mse:.3e} cos={avg_cos:.6f}", flush=True)

    # 4. validate vs stored baseline
    print("[4/5] validating ...", flush=True)
    with torch.no_grad():
        q_logits = model(**enc).logits.float()
    logit_mse = (base_logits - q_logits).pow(2).mean().item()
    logit_cos = F.cosine_similarity(base_logits.flatten(), q_logits.flatten(), dim=0).item()
    print(f"  logits MSE={logit_mse:.3e} cosine={logit_cos:.6f}", flush=True)

    corpus = [
        "The mitochondria generate adenosine triphosphate through oxidative phosphorylation.",
        "Beta blockers antagonize beta adrenergic receptors, reducing heart rate and contractility.",
    ]
    def ppl():
        nll, nt = 0.0, 0
        with torch.no_grad():
            for text in corpus:
                e = tok(text, return_tensors="pt")
                out = model(**e, labels=e["input_ids"])
                n = int(e["input_ids"].numel())
                nll += float(out.loss) * n
                nt += n
        return torch.exp(torch.tensor(nll / nt)).item()
    # baseline ppl needs unpatched model: approximate with pre-patch run is gone,
    # so report quantized ppl + note. (Reloading baseline doubles time; skip by design.)
    ppl_q = ppl()
    print(f"  quantized ppl={ppl_q:.3f} (baseline ppl needs separate run)", flush=True)

    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0], skip_special_tokens=True)
    print(f"  generation: {gen[:200]}...", flush=True)

    report = {
        "model_id": args.model_id, "total_params": total_params,
        "lattice": args.lattice, "stc_frac": args.stc_frac,
        "n_linears": n_mod, "quant_params": tot_w, "residuals": tot_res,
        "weight_mse": avg_mse, "weight_cos": avg_cos,
        "logit_mse": logit_mse, "logit_cos": logit_cos,
        "quant_ppl": ppl_q, "generation": gen, "quant_time_s": round(dt, 1),
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[5/5] wrote {args.report}", flush=True)

    if args.save_dequant:
        model.save_pretrained(args.save_dequant, safe_serialization=True)
        tok.save_pretrained(args.save_dequant)
        print(f"saved dequantized checkpoint to {args.save_dequant}", flush=True)


if __name__ == "__main__":
    main()
