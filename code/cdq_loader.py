"""Minimal CDQ-LRR v1 loader: read qwen05.cdq and patch a transformers model.

Usage:
    python cdq_loader.py --model . --prompt "Hello world"

Requires: torch, transformers, safetensors. No training code, no calibration.
"""
import argparse
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC, VER = b"CDQ1", 1
TILE = 32
LATTICE = torch.tensor([
    -1.0000, -0.7321, -0.5284, -0.3752, -0.2612, -0.1774, -0.1167, -0.0734,
    -0.0432, -0.0229, -0.0101, -0.0031, -0.0005, 0.0005, 0.0031, 0.0101,
    0.0229, 0.0432, 0.0734, 0.1167, 0.1774, 0.2612, 0.3752, 0.5284,
    0.7321, 1.0000,
], dtype=torch.bfloat16)


def _u5(buf: bytes):
    acc = int.from_bytes(buf, "little")
    return [(acc >> (5 * i)) & 0x1F for i in range(32)]


def _res(buf: bytes, n: int):
    if not n:
        return [], []
    acc = int.from_bytes(buf, "little")
    offs, vals, pos = [], [], 0
    for _ in range(n):
        offs.append((acc >> pos) & 0x1F)
        pos += 5
        raw = (acc >> pos) & 0xFFFF
        pos += 16
        vals.append(torch.tensor(raw, dtype=torch.uint16).view(torch.bfloat16))
    return offs, vals


class CDQLinear(nn.Module):
    def __init__(self, q, s, offs, vals, shape, bias=None):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("s", s)
        self.register_buffer("offs", offs)
        self.register_buffer("vals", vals)
        self.shape = tuple(shape)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x):
        w = LATTICE.to(x.device)[self.q.to(x.device).long()].to(torch.bfloat16) * self.s.to(x.device)
        if int(self.offs.numel()):
            w.view(-1)[self.offs.to(x.device)] = self.vals.to(torch.bfloat16).to(x.device)
        w = w.view(self.shape).to(x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)


def load_cdq(model, path="qwen05.cdq"):
    raw = open(path, "rb").read()
    p = 0
    assert raw[:4] == MAGIC
    ver, n = struct.unpack_from("<HH", raw, 4)
    assert ver == VER
    p = 8 + 52
    got = 0
    for _ in range(n):
        nl = struct.unpack_from("<H", raw, p)[0]
        p += 2
        name = raw[p:p + nl].decode()
        p += nl
        rows, cols, nt = struct.unpack_from("<III", raw, p)
        p += 12
        F8 = torch.float8_e4m3fn
        # parse per-tile records
        tiles = []
        for _ in range(nt):
            sc, tc = raw[p], raw[p + 1]
            p += 2
            qi = _u5(raw[p:p + 20])
            p += 20
            nb = (tc * 21 + 7) // 8
            offs, vals = _res(raw[p:p + nb], tc)
            p += nb
            tiles.append((sc, tc, qi, offs, vals))
        # attach
        modname = name.replace(".weight", "")
        try:
            mod = model.get_submodule(modname)
        except AttributeError:
            continue
        q = torch.tensor([qi for _, _, qi, _, _ in tiles], dtype=torch.uint8).view(-1, 32)
        f8 = torch.tensor([t[0] for t in tiles], dtype=torch.uint8).view(-1, 1)
        sigma = f8.view(F8).to(torch.bfloat16)
        offs_all, vals_all, base = [], [], 0
        for ti, (_, _, _, offs, vals) in enumerate(tiles):
            offs_all += [ti * 32 + o for o in offs]
            vals_all += vals
        cdq = CDQLinear(q, sigma,
                        torch.tensor(offs_all, dtype=torch.long),
                        torch.stack(vals_all) if vals_all else torch.empty(0, dtype=torch.bfloat16),
                        (rows, cols),
                        bias=mod.bias.data if mod.bias is not None else None)
        parent, _, attr = modname.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, attr, cdq)
        got += 1
    print(f"patched {got} CDQLinear modules from {path}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".")
    ap.add_argument("--cdq", default="qwen05.cdq")
    ap.add_argument("--prompt", default="Explain the mechanism of action of beta blockers in 2 sentences.")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True).eval()
    load_cdq(m, args.cdq)
    enc = tok(args.prompt, return_tensors="pt")
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
