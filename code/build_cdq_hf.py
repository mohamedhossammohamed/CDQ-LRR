"""Build HF-format checkpoint carrying CDQ-LRR v1 weights.

Loads models/qwen-local (BF16), applies CDQ-LRR 26-state / STC 1.56% to the
168 projection matrices, writes the *dequantized* weights (BF16, carrying the
quantization error) as a standard transformers checkpoint. Norms, embeddings
and lm_head are untouched originals.

Output: models/qwen05-cdq-hf/ (config + tokenizer + model.safetensors)
"""
import argparse
import shutil
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdq_lrr_qwen05 import (CAMKII_LATTICE, quantize_cdq_lrr, dequantize,
                            TARGET_SUBSTRINGS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="HF checkpoint dir (default: $CDQ_MODEL or models/qwen-local)")
    ap.add_argument("--dst", default="models/qwen05-cdq-hf")
    args = ap.parse_args()
    import os
    SRC = args.src or os.environ.get("CDQ_MODEL", "models/qwen-local")
    DST = args.dst
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("loading baseline ...", flush=True)
    tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        SRC, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True).eval()
    n = 0
    with torch.no_grad():
        for name, mod in m.named_modules():
            if isinstance(mod, nn.Linear) and any(s in name for s in TARGET_SUBSTRINGS):
                pack = quantize_cdq_lrr(mod.weight.data, lattice=CAMKII_LATTICE,
                                        stc_frac=0.0156)
                mod.weight.data = dequantize(pack).to(torch.bfloat16)
                n += 1
                if n % 40 == 0:
                    print(f"  {n} ...", flush=True)
    print(f"dequantized {n} projections; saving to {DST} ...", flush=True)
    m.save_pretrained(DST, safe_serialization=True)
    tok.save_pretrained(DST)
    # keep generation config alongside
    try:
        shutil.copy(f"{SRC}/generation_config.json", f"{DST}/generation_config.json")
    except OSError:
        pass
    print("done", flush=True)


if __name__ == "__main__":
    main()
