"""Minimal HF safetensors -> GGUF (qwen2 arch, F16) for Qwen2.5-0.5B-CDQ.

Uses the gguf python package only. Includes q/k/v biases, tied-output
duplicated, and gpt2-style BPE tokenizer from tokenizer.json.
Then: llama-quantize out.gguf Q8_0 (or Q4_K_M) for Ollama.
"""
import json
import sys
import numpy as np
import torch
from safetensors import safe_open

SRC = "/Users/mohammedhossam/Desktop/MZSAE/models/qwen05-cdq-hf"
OUT = "/Users/mohammedhossam/Desktop/MZSAE/models/qwen05-cdq-f16.gguf"


def main():
    import gguf
    cfg = json.load(open(f"{SRC}/config.json"))
    assert cfg["model_type"] == "qwen2"
    tokj = json.load(open(f"{SRC}/tokenizer.json"))

    w = gguf.GGUFWriter(OUT, "qwen2")
    w.add_quantization_version(2)
    w.add_file_type(1)  # MOSTLY_F16
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_block_count(cfg["num_hidden_layers"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_rope_dimension_count(cfg["hidden_size"] // cfg["num_attention_heads"])
    w.add_head_count(cfg["num_attention_heads"])
    w.add_head_count_kv(cfg["num_key_value_heads"])
    w.add_key_length(cfg["hidden_size"] // cfg["num_attention_heads"])
    w.add_value_length(cfg["hidden_size"] // cfg["num_attention_heads"])
    w.add_rope_freq_base(cfg.get("rope_theta", 1000000.0))
    w.add_layer_norm_rms_eps(cfg.get("rms_norm_eps", 1e-6))

    # tokenizer (gpt2 BPE)
    vocab = tokj["model"]["vocab"]
    ids = sorted(vocab, key=vocab.get)
    # Qwen2: 151643 BPE + added specials + implicit reserved controls = 151936
    by_id = {a["id"]: a["content"] for a in tokj.get("added_tokens", [])}
    full = list(ids)
    for i in range(len(full), cfg["vocab_size"]):
        full.append(by_id.get(i, f"<|reserved_{i}|>"))
    assert len(full) == cfg["vocab_size"]
    toks = [t.encode("utf-8", errors="replace") if isinstance(t, str) else t for t in full]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(toks)
    import gguf as _g
    NORMAL, CONTROL = 1, 3
    w.add_token_types([NORMAL if i < len(ids) else CONTROL for i in range(len(full))])
    merges = tokj["model"]["merges"]
    if merges and isinstance(merges[0], (list, tuple)):
        merges = [" ".join(p) for p in merges]
    w.add_token_merges(merges)
    added = tokj.get("added_tokens", [])
    bos = next((t["id"] for t in added if t["content"] in ("<|im_start|>", "<s>")), cfg.get("bos_token_id", 151643))
    eos = next((t["id"] for t in added if t["content"] in ("<|im_end|>", "</s>")), cfg.get("eos_token_id", 151645))
    w.add_bos_token_id(bos)
    w.add_eos_token_id(eos)
    w.add_unk_token_id(0)
    try:
        w.add_chat_template(open(f"{SRC}/chat_template.jinja").read())
    except OSError:
        pass

    f = safe_open(f"{SRC}/model.safetensors", framework="pt")

    def np16(key):
        return f.get_tensor(key).to(torch.float16).numpy()

    def np32(key):
        return f.get_tensor(key).to(torch.float32).numpy()

    def add(name, key, npfn=None):
        t = f.get_tensor(key)
        # llama.cpp CPU backend requires all 1-D tensors (norms, biases) in F32
        fn = np32 if (npfn is None and t.dim() == 1) else (npfn or np16)
        w.add_tensor(name, fn(key))

    add("token_embd.weight", "model.embed_tokens.weight")
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        add(f"blk.{i}.attn_norm.weight", f"{p}.input_layernorm.weight")
        add(f"blk.{i}.attn_q.weight", f"{p}.self_attn.q_proj.weight")
        add(f"blk.{i}.attn_k.weight", f"{p}.self_attn.k_proj.weight")
        add(f"blk.{i}.attn_v.weight", f"{p}.self_attn.v_proj.weight")
        for suffix, gg in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v")):
            try:
                add(f"blk.{i}.{gg}.bias", f"{p}.self_attn.{suffix}.bias", np32)
            except Exception as e:
                print(f"  no bias {p}.{suffix}: {e}")
        add(f"blk.{i}.attn_output.weight", f"{p}.self_attn.o_proj.weight")
        add(f"blk.{i}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight")
        add(f"blk.{i}.ffn_gate.weight", f"{p}.mlp.gate_proj.weight")
        add(f"blk.{i}.ffn_up.weight", f"{p}.mlp.up_proj.weight")
        add(f"blk.{i}.ffn_down.weight", f"{p}.mlp.down_proj.weight")
    add("output_norm.weight", "model.norm.weight")
    try:
        add("output.weight", "lm_head.weight")
    except Exception:
        add("output.weight", "model.embed_tokens.weight")  # tied embeddings
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
