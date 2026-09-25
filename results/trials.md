# CDQ-LRR trial log — Qwen2.5-0.5B-Instruct (all measured, Sept 2026)

Base model: `Qwen/Qwen2.5-0.5B-Instruct`, BF16, 24 layers, hidden 896.
Quantized scope: 168 projection matrices (`q/k/v/o/gate/up/down_proj`) =
**357,826,560 params / 11,182,080 tiles of 32**. Norms, embeddings, `lm_head` stay BF16.
Hardware: Apple Silicon (CPU/PyTorch), kent.

## Configurations

| # | Lattice | STC | Artifact | Size | BPW |
|---|---|---|---|---|---|
| T1 (v1, production) | 26-state, 5-bit | 1.56% (5,582,210) | `qwen05.cdq` | 261.5 MB | 5.847 |
| T2 (draft) | 16-state Lloyd-Max, 4-bit nibble | 1.56% | `qwen05.cdq` (superseded) | 216.8 MB | 4.847 |
| T3 (draft-zero) | 16-state zero-clamped, 4-bit | 1.56% | — | 216.8 MB | 4.847 |
| T4 (sweep) | 16-state zero-clamped, 4-bit | 3.125% (11,182,127, 1.00/tile) | — | 232.1 MB | 5.189 |

BPW anatomy (measured): `index_bits + 0.250 (FP8 σ) + residuals·21b + 0.250 (tag byte) + header`.

## Weight-space fidelity (avg over 168 matrices)

| Config | MSE | Cosine |
|---|---|---|
| T1 | 3.315e-06 | 0.996393 |
| T2 | 3.279e-06 | 0.996837 |
| T3 | 3.749e-06 | 0.996137 |
| T4 | 3.522e-06 | 0.996341 |

## Logit-level (prompt: beta blockers, greedy 60)

| Config | Logit MSE | Logit cosine | Perplexity (4-sent) |
|---|---|---|---|
| BF16 baseline | — | 1.0000 | 26.974 |
| T1 | 0.643 | **0.9819** | **26.256** (Δ −0.72) |
| T2 | 1.966 | 0.9459 | 41.050 |
| T3 | 1.819 | 0.9485 | 36.547 |
| T4 | 1.713 | 0.9514 | 35.651 |

## Smartness bench — 48 hand-written MCQs, letter-logprob scoring

`results/cdq_smartness.json` (runnable: `code/cdq_smartness_bench.py`)

| Model | Score | Flips vs BF16 | McNemar (discordant) |
|---|---|---|---|
| BF16 | 29/48 (60.4%) | — | — |
| T1 | 23/48 (47.9%) | 15 | 8 bad / 2 good, p≈0.055 |
| T3-equiv (16-zero) | 21/48 (43.8%) | 20 | 12 bad / 4 good, p≈0.038 |

Pattern: factual recall intact (science 8/10, commonsense 6/8 identical under T1);
symbolic reasoning drops first (arithmetic 4→2, logic 4→2). n=48: indicative, not definitive.

## Distribution finding (kills the 3.35-bit assumption)

Index histogram over all 357.8M assignments (26-state): inner `q9..q16` = **6.9%**
(not >70%); peaks at `q3/q22` (~7.7%); entropy = **4.338 bits**. Per-32 max
normalization spreads mass to mid-lattice, so nibble+escape costs 5.73 BPW —
worse than flat 5.00. Measured, not assumed.

## Lattices

26-state (v1): `-1.0, -0.7321, -0.5284, -0.3752, -0.2612, -0.1774, -0.1167,
-0.0734, -0.0432, -0.0229, -0.0101, -0.0031, -0.0005, 0.0005, 0.0031, 0.0101,
0.0229, 0.0432, 0.0734, 0.1167, 0.1774, 0.2612, 0.3752, 0.5284, 0.7321, 1.0`

16-state zero-clamped (Qwen-fitted, draft): `-0.9685, -0.7525, -0.5834, -0.4414,
-0.3174, -0.2048, -0.0998, 0.0, 0.0998, 0.2048, 0.3174, 0.4414, 0.5834,
0.7525, 0.9685, 0.9685(dup)`

## Key inference (T4)

Doubling residuals (5.58M→11.18M, +0.34 BPW) moved logit cosine only
0.9485→0.9514: the 4-bit shortfall is **bulk-resolution-limited** (14 non-zero
centroids over 4,864-dim dots), not outlier coverage. 26-state/5-bit is the
proven minimum for this model scale.
