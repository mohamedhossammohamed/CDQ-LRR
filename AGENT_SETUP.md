# Plug-and-play prompt for a coding agent (Cursor / Claude Code / Codex / Opencode)

Copy everything below the line into the agent, filling in the two blanks.
Repo: https://github.com/mohamedhossammohamed/CDQ-LRR

---

Clone https://github.com/mohamedhossammohamed/CDQ-LRR and run the CDQ-LRR
post-training quantization trial on the Hugging Face model
__MODEL_ID__ (default if I leave this blank: XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B).
Save all logs under __LOG_DIR__ (default: ./logs/).

Do exactly this, in order. Stop and report an error instead of guessing if any
step fails:

1. `pip install -r requirements.txt` (needs torch, transformers, safetensors,
   huggingface_hub, numpy). The model in step 2 may need a very new
   `transformers` (qwen3_5 arch) — upgrade with `pip install -U transformers`
   if the load fails with an architecture error.
2. Download: `huggingface-cli download __MODEL_ID__` — or let the runner do it.
   It is ~19 GB; resume with `-c` behavior is automatic via snapshot_download.
3. Run: `python code/run_any_model.py --model-id __MODEL_ID__ --lattice 26
   --stc-frac 0.0156 --report __LOG_DIR__/cdq_trial_report.json`.
   This is CPU-only, no GPU needed for the quant pass. Peak RAM ≈ 2x model
   size. It takes roughly 1–3 minutes per billion params for quantization.
4. Verify these gates and report each explicitly:
   - weight cosine average ≥ 0.996 and weight MSE ~1e-6 (printed by the runner)
   - logit cosine vs baseline ≥ 0.97 on the built-in prompt
   - greedy generation is fluent and on-topic (paste the full text)
5. If logit cosine lands below 0.97, do NOT tune anything — that is itself the
   result. Rerun once with `--stc-frac 0.03125` and report whether it moved.
   If it barely moves (like our 0.948→0.951 on 0.5B/4-bit), the config is
   bulk-limited: stop and report.
6. Send back: the full stdout log, `cdq_trial_report.json`, total wall time,
   and machine specs (RAM, CPU, GPU if any).

Do not upload weights anywhere and do not use any API keys except a
user-supplied HF_TOKEN env var for gated models. Do not modify the method —
measure it.
