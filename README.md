# CDQ-LRR — CaMKII-Engineered Dendritic Quantal Compression with Lossless Residual Routing

A biophysics-inspired post-training quantization method for LLMs, **implemented,
measured, and honestly reported** on Qwen2.5-0.5B-Instruct.

- 📄 Theory + full math: [`THEORY_FULL.md`](THEORY_FULL.md)
- 📊 Measured trial log: [`results/trials.md`](results/trials.md)
- 🤗 Quantized weights: [aws3712/Qwen2.5-0.5B-CDQ-LRR](https://huggingface.co/aws3712/Qwen2.5-0.5B-CDQ-LRR)
  (safetensors checkpoint · `qwen05.cdq` native pack · Q8_0 GGUF · Modelfile · loader)
- 🖥️ Run on Ollama: `ollama create cdq-qwen05 -f Modelfile && ollama run cdq-qwen05 "..."`
- 🌐 Project page: GitHub Pages site in [`docs/`](docs/) (Anthropic-style research page)

## The method in 30 seconds

1. **Tile** each projection into 32-element micro-tiles; per-tile scale `σ = max|w|` in FP8.
2. **Tag** the top 1.56% salient weights (`|w|·(tile_var+ε)`) — stored bit-exact BF16.
3. **Snap** the rest to a 26-state quasi-log lattice (5-bit index, 52-byte register LUT).
4. **Decode** on the fly: `ŵ = σ·LUT[q]`, inject residuals, straight into GEMV.

Measured on 357.8M params: **5.85 BPW, 0.9964 weight cosine, 0.9819 logit cosine.**

## What the trials actually showed

| Claim (theory) | Measurement |
|---|---|
| 3.94 BPW net | **5.85 BPW** — index entropy is 4.34 bits, not 3.35 (normalization spreads mass mid-lattice) |
| 100% reasoning retention | **23/48 vs 29/48** on a 48-item MCQ bench; factual recall intact, symbolic reasoning drops first |
| 4-bit nibble path viable | **Rejected for quality**: logit cosine stalls at ~0.95 even with doubled residuals — bulk-limited |

The 16-state/4-bit tier survives as a **draft/speculative tier** (4.85 BPW, native nibble ALU),
not as the quality tier. See [`results/trials.md`](results/trials.md) for every number.

## Larger models — status: NOT YET TESTED (read this before citing)

All results above are **Qwen2.5-0.5B-Instruct only**. The 70B-class projections in
`THEORY_FULL.md` are extrapolations, not measurements. A 9B validation
(MiMo-V2.6-Distill-Qwen-9B, 18.8 GB) is downloading — on a very slow connection in
Egypt, shard 1 of 4 has taken over a day so far. Prior reasons for cautious optimism
(local per-tile statistics, ~1–2% outlier fractions reported across scales) and one
reason for caution (bulk error compounds with depth — already visible at 24 layers)
are laid out on the project page. Please do not cite large-model numbers until the
9B trial lands.

## Reproduce

```bash
# quantize + validate (needs torch, transformers, safetensors; local Qwen2.5-0.5B)
python code/cdq_lrr_qwen05.py --lattice 26 --stc-frac 0.0156
# pack native .cdq
python code/cdq_pack.py --states 26 --stc-frac 0.0156 --out qwen05.cdq
# smartness bench
python code/cdq_smartness_bench.py
# HF checkpoint + GGUF + upload
python code/build_cdq_hf.py && python code/hf_to_gguf_qwen2.py
```

Layout: `code/` runnable pipeline · `csrc/` C++/Metal kernels · `results/` logs +
`Modelfile` + `cdq_loader.py` · `docs/` GitHub Pages site.
