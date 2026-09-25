"""CDQ-LRR (CaMKII-Engineered Dendritic Quantal Compression with Lossless Residual Routing)
on Qwen2.5-0.5B-Instruct, local BF16 weights.

Pipeline:
  1. 32-element dendritic micro-tiles on 2D projections
     (q/k/v/o/gate/up/down_proj). Norms, embeddings, lm_head stay BF16.
  2. Astrocytic divisive normalization: sigma=max|w|, FP8(E4M3) quant, x=w/sigma
  3. STC: saliency=|w|*(tile_var+1e-6), top-1.56% layer-wise -> sparse BF16 residuals
  4. CaMKII 26-state nearest-centroid map (5-bit index)
  5. CDQLinear: on-the-fly dequant + residual injection, validation
     (MSE, cosine, perplexity, generation).

Local model: models/qwen-local (942MB BF16 safetensors).
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Override with:  --model /path/to/hf-checkpoint   or   CDQ_MODEL=/path/to/...
MODEL_DIR = os.environ.get("CDQ_MODEL", "models/qwen-local")
TARGET_SUBSTRINGS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TILE = 32
STC_FRAC = 0.0156  # top 1.56%

CAMKII_LATTICE = torch.tensor([
    -1.0000, -0.7321, -0.5284, -0.3752, -0.2612, -0.1774, -0.1167, -0.0734,
    -0.0432, -0.0229, -0.0101, -0.0031, -0.0005,  0.0005,  0.0031,  0.0101,
     0.0229,  0.0432,  0.0734,  0.1167,  0.1774,  0.2612,  0.3752,  0.5284,
     0.7321,  1.0000,
], dtype=torch.bfloat16)

# Zero-clamped 15-state symmetric lattice in 16 nibble slots (q15 = dup max).
# Fit: abs-domain Lloyd-Max with centroid fixed at 0 on 2M normalized Qwen
# samples. q7 = basal inactive (exact 0): |x| < 0.0499 prunes to zero instead
# of injecting +/-0.0458 noise. Normalized MSE 0.001355 (vs 0.001189 no-zero;
# the downstream bet is logit-level, see validation).
CAMKII_LATTICE_16 = torch.tensor([
    -0.9685, -0.7525, -0.5834, -0.4414, -0.3174, -0.2048, -0.0998,  # q0..q6
     0.0000,                                                        # q7 zero
     0.0998,  0.2048,  0.3174,  0.4414,  0.5834,  0.7525,  0.9685,  # q8..q14
     0.9685,                                                        # q15 dup
], dtype=torch.bfloat16)

FP8_DTYPE = torch.float8_e4m3fn


def fp8_quantize_scale(sigma_bf16: torch.Tensor) -> torch.Tensor:
    """Quantize per-tile scale to FP8 E4M3 then back to BF16 for compute."""
    try:
        return sigma_bf16.to(FP8_DTYPE).to(torch.bfloat16)
    except Exception:
        # Simulated FP8 E4M3: clamp to max 448, stochastic-free round to ~3 mantissa bits
        s = sigma_bf16.float().clamp(min=1e-8, max=448.0)
        # keep sign (scales are positive) — emulate 4-exp/3-mantissa by log2 rounding
        log2 = torch.floor(torch.log2(s) + 0.5)
        mant = s / (2.0 ** log2)  # in [0.7, 1.4)
        mant_q = torch.round((mant - 1.0) * 8.0) / 8.0 + 1.0
        return (mant_q * (2.0 ** log2)).clamp(min=1e-8).to(torch.bfloat16)


def quantize_cdq_lrr(weight: torch.Tensor, chunk_tiles: int = 8192, lattice=None,
                     stc_frac: float = None):
    """Full CDQ-LRR quantize of a 2D BF16 weight matrix.

    lattice: centroid table (default 26-state CAMKII_LATTICE).
    Returns dict with quant_indices (uint8 [N,32]), scales_f8 (fp8 [N,1]),
    residual_indices (int64 [R]), residual_values (bf16 [R]), orig_shape.
    """
    assert weight.dim() == 2, f"expected 2D, got {weight.shape}"
    if lattice is None:
        lattice = CAMKII_LATTICE
    if stc_frac is None:
        stc_frac = STC_FRAC
    orig_shape = tuple(weight.shape)
    device = weight.device
    flat = weight.detach().to(torch.bfloat16).contiguous().view(-1, TILE)
    n_tiles = flat.shape[0]
    lat = lattice.to(device)

    # Phase 1: sigma per tile
    sigma = torch.max(torch.abs(flat), dim=1, keepdim=True).values.clamp(min=1e-8)
    sigma_q = fp8_quantize_scale(sigma)  # BF16 view of FP8 scale [N,1]

    # Phase 2: saliency + top-1.56% mask (layer-wise dynamic)
    tile_var = torch.var(flat.float(), dim=1, keepdim=True)  # [N,1]
    saliency = torch.abs(flat.float()) * (tile_var + 1e-6)  # [N,32]
    k = max(1, int(round(saliency.numel() * stc_frac)))
    thresh = torch.quantile(saliency.flatten(), 1.0 - stc_frac)
    outlier_mask = saliency >= thresh  # bool [N,32]
    # exact count guard (quantile ties can overshoot)
    if int(outlier_mask.sum()) > k * 2:  # severe ties: take true top-k
        _, topk_idx = torch.topk(saliency.flatten(), k, sorted=False)
        outlier_mask = torch.zeros_like(saliency, dtype=torch.bool)
        outlier_mask.view(-1)[topk_idx] = True

    res_flat_idx = torch.where(outlier_mask.view(-1))[0].to(torch.int64)
    res_values = flat.view(-1)[res_flat_idx].to(torch.bfloat16).contiguous()

    # Phase 3+4: chunked nearest-centroid + reconstruction (avoid [N,32,26] OOM)
    quant_indices = torch.empty((n_tiles, TILE), dtype=torch.uint8, device=device)
    recon = torch.empty_like(flat)
    for s in range(0, n_tiles, chunk_tiles):
        e = min(s + chunk_tiles, n_tiles)
        norm = flat[s:e].float() / sigma_q[s:e].float()  # [C,32] fp32 for distance
        d = torch.abs(norm.unsqueeze(-1) - lat.float())  # [C,32,K]
        q = torch.argmin(d, dim=-1).to(torch.uint8)  # [C,32]
        quant_indices[s:e] = q
        recon[s:e] = lat[q.long()].to(torch.bfloat16) * sigma_q[s:e]

    # Lossless residual injection (bit-exact restore)
    recon.view(-1)[res_flat_idx] = flat.view(-1)[res_flat_idx]

    # Store scales as true FP8 tensor for accounting + BF16 view for compute
    try:
        scales_f8 = sigma.to(FP8_DTYPE).contiguous()  # [N,1] fp8 storage
    except Exception:
        scales_f8 = sigma_q.to(torch.bfloat16).contiguous()

    return {
        "quant_indices": quant_indices.cpu().contiguous(),
        "scales_f8": scales_f8.cpu().contiguous(),
        "scales_bf16": sigma_q.cpu().contiguous(),
        "residual_indices": res_flat_idx.cpu().contiguous(),
        "residual_values": res_values.cpu().contiguous(),
        "orig_shape": orig_shape,
        "outlier_count": int(res_flat_idx.numel()),
        "numel": int(flat.numel()),
        "n_tiles": int(n_tiles),
        "lattice": lattice.cpu().contiguous(),
        "n_states": int(lattice.numel()),
    }


def _pack_lattice(pack):
    lat = pack.get("lattice", None)
    return CAMKII_LATTICE if lat is None else lat


def dequantize(pack, device=None, dtype=torch.bfloat16):
    """Reconstruct BF16 weight from pack: w_hat = sigma * Lattice[q] + residuals."""
    lat = _pack_lattice(pack)
    lattice = lat if device is None else lat.to(device)
    q = pack["quant_indices"] if device is None else pack["quant_indices"].to(device)
    s = pack["scales_bf16"] if device is None else pack["scales_bf16"].to(device)
    w = lattice[q.long()].to(torch.bfloat16) * s  # [N,32]
    ri = pack["residual_indices"] if device is None else pack["residual_indices"].to(device)
    rv = pack["residual_values"] if device is None else pack["residual_values"].to(device)
    if int(ri.numel()) > 0:
        w.view(-1)[ri.long()] = rv.to(torch.bfloat16)
    return w.view(pack["orig_shape"]).to(dtype)


class CDQLinear(nn.Module):
    """Linear with on-the-fly CDQ-LRR dequantization + sparse residual routing."""

    def __init__(self, pack, in_features: int, out_features: int, bias=None):
        super().__init__()
        self.register_buffer("quant_indices", pack["quant_indices"])
        # scales_f8 may be fp8 or bf16 fallback — keep as stored
        self.register_buffer("scales_f8", pack["scales_f8"])
        self.register_buffer("scales_bf16", pack["scales_bf16"])
        self.register_buffer("residual_indices", pack["residual_indices"])
        self.register_buffer("residual_values", pack["residual_values"])
        self.register_buffer("lattice", _pack_lattice(pack).contiguous())
        self.orig_shape = tuple(pack["orig_shape"])
        self.in_features = in_features
        self.out_features = out_features
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().to(torch.bfloat16), requires_grad=False)
        else:
            self.bias = None
        self._pack_meta = {
            "outlier_count": pack["outlier_count"],
            "numel": pack["numel"],
            "n_tiles": pack["n_tiles"],
        }

    @classmethod
    def from_linear(cls, lin: nn.Linear, lattice=None):
        with torch.no_grad():
            pack = quantize_cdq_lrr(lin.weight.data, lattice=lattice)
        obj = cls(pack, lin.in_features, lin.out_features,
                  bias=lin.bias.data if lin.bias is not None else None)
        return obj, pack

    def dequant_weight(self):
        q = self.quant_indices.to(self.scales_bf16.device)
        lat = self.lattice.to(self.scales_bf16.device)
        w = lat[q.long()].to(torch.bfloat16) * self.scales_bf16
        if int(self.residual_indices.numel()) > 0:
            w.view(-1)[self.residual_indices.long()] = self.residual_values.to(torch.bfloat16)
        return w.view(self.orig_shape)

    def forward(self, x):
        w = self.dequant_weight().to(x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

    def extra_repr(self):
        m = self._pack_meta
        return (f"shape={self.orig_shape}, states={int(self.lattice.numel())}, "
                f"outliers={m['outlier_count']}/{m['numel']} "
                f"({100*m['outlier_count']/max(1,m['numel']):.2f}%)")


def effective_bpw(total_numel, total_tiles, total_residuals, index_bits=5,
                  header_bytes_per_tile=0):
    # index_bits per weight + 8-bit scale per 32 weights
    # + residual: 16-bit value + 32-bit coord (prototype accounting)
    # + optional per-tile header bytes
    bits = (total_numel * index_bits + total_tiles * 8
            + total_residuals * (16 + 32)
            + total_tiles * header_bytes_per_tile * 8)
    return bits / max(1, total_numel)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--prompt", default="Explain the mechanism of action of beta blockers in 2 sentences.")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--ppl-texts", type=int, default=4)
    ap.add_argument("--lattice", type=int, choices=(26, 16), default=26,
                    help="CaMKII lattice states: 26 (5-bit) or 16 (4-bit Lloyd-Max)")
    ap.add_argument("--stc-frac", type=float, default=STC_FRAC,
                    help="STC outlier fraction (0.0156 = top 1.56%, 0.03125 = 1/tile)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    lattice = CAMKII_LATTICE if args.lattice == 26 else CAMKII_LATTICE_16
    index_bits = 5 if args.lattice == 26 else 4
    print(f"CDQ-LRR lattice: {args.lattice}-state ({index_bits}-bit indices), "
          f"STC top {100*args.stc_frac:.3f}%")

    print(f"Loading tokenizer + BF16 baseline from {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
    )
    base.eval()

    # Collect target linears
    targets = [(n, m) for n, m in base.named_modules()
               if isinstance(m, nn.Linear) and any(s in n for s in TARGET_SUBSTRINGS)]
    print(f"Found {len(targets)} projection matrices for CDQ-LRR "
          f"(lm_head/norms/embeddings stay BF16).")

    # Quantize (weight-space metrics)
    packs, mse_list, cos_list = {}, [], []
    t0 = time.time()
    tot_numel = tot_tiles = tot_res = 0
    for i, (name, mod) in enumerate(targets):
        ow = mod.weight.data.to(torch.bfloat16)
        pack = quantize_cdq_lrr(ow, lattice=lattice, stc_frac=args.stc_frac)
        packs[name] = pack
        rw = dequantize(pack).to(torch.bfloat16)
        mse = (ow.float() - rw.float()).pow(2).mean().item()
        cos = F.cosine_similarity(ow.flatten().float(), rw.flatten().float(), dim=0).item()
        mse_list.append(mse)
        cos_list.append(cos)
        tot_numel += pack["numel"]
        tot_tiles += pack["n_tiles"]
        tot_res += pack["outlier_count"]
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(targets)}] {name}: mse={mse:.3e} cos={cos:.6f} "
                  f"outliers={pack['outlier_count']}/{pack['numel']}")
    print(f"Quantized {len(targets)} mats in {time.time()-t0:.1f}s | "
          f"avg MSE={sum(mse_list)/len(mse_list):.3e} | "
          f"avg cos={sum(cos_list)/len(cos_list):.6f}")
    print(f"Effective bitrate ~ {effective_bpw(tot_numel, tot_tiles, tot_res, index_bits=index_bits):.3f} bpw "
          f"({index_bits}b lattice + FP8 scales + {tot_res} BF16 residuals)")

    # Patch a second model instance with CDQLinear (keep baseline intact)
    print("\nPatching quantized model with CDQLinear modules ...")
    quant_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
    )
    quant_model.eval()
    for name, mod in list(quant_model.named_modules()):
        if isinstance(mod, nn.Linear) and any(s in name for s in TARGET_SUBSTRINGS):
            parent_path, _, attr = name.rpartition(".")
            parent = quant_model.get_submodule(parent_path) if parent_path else quant_model
            pack = packs[name]
            cdq = CDQLinear(pack, mod.in_features, mod.out_features,
                            bias=mod.bias.data if mod.bias is not None else None)
            setattr(parent, attr, cdq)
    n_cdq = sum(1 for _ in quant_model.modules() if isinstance(_, CDQLinear))
    print(f"Patched {n_cdq} CDQLinear modules.")

    # Logit-level validation on the test prompt
    print("\n--- Logit-level validation ---")
    enc = tok(args.prompt, return_tensors="pt")
    with torch.no_grad():
        lb = base(**enc).logits.float()
        lq = quant_model(**enc).logits.float()
    logit_mse = (lb - lq).pow(2).mean().item()
    logit_cos = F.cosine_similarity(lb.flatten(), lq.flatten(), dim=0).item()
    print(f"prompt: {args.prompt!r}")
    print(f"logits MSE={logit_mse:.3e} cosine={logit_cos:.6f}")

    # Perplexity on small fixed corpus (offline, no download)
    corpus = [
        "The mitochondria generate adenosine triphosphate through oxidative phosphorylation.",
        "Beta blockers antagonize beta adrenergic receptors, reducing heart rate and contractility.",
        "Calcium/calmodulin-dependent protein kinase II forms a dodecameric holoenzyme in neurons.",
        "Quantization compresses neural network weights to lower bit widths for efficient inference.",
    ][:args.ppl_texts]

    def ppl(model):
        nll_sum, ntok = 0.0, 0
        with torch.no_grad():
            for text in corpus:
                e = tok(text, return_tensors="pt")
                out = model(**e, labels=e["input_ids"])
                # out.loss is mean NLL over tokens
                n = int(e["input_ids"].numel())
                nll_sum += float(out.loss) * n
                ntok += n
        return torch.exp(torch.tensor(nll_sum / ntok)).item()

    ppl_b = ppl(base)
    ppl_q = ppl(quant_model)
    print(f"perplexity baseline={ppl_b:.3f} cdq-lrr={ppl_q:.3f} "
          f"(delta={ppl_q-ppl_b:+.3f})")

    # Generation test (greedy)
    print("\n--- Live generation (greedy) ---")
    enc = tok(args.prompt, return_tensors="pt")
    with torch.no_grad():
        for label, model in (("BF16 baseline", base), ("CDQ-LRR", quant_model)):
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            print(f"[{label}] {tok.decode(out[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
