"""CDQ-LRR production packer: tile-local STC + 5-bit bitpack + .cdq codec.

Findings from measured histogram (357.85M indices, Qwen2.5-0.5B-Instruct):
  - inner q9..q16 = 6.9% (NOT >70%). Distribution peaks at mid-lattice
    (q3/q22 ~7.7%), valley at center. Entropy = 4.34 bits.
  - => nibble+escape (4b base + 32b mask + 4b ext) costs ~5.73 BPW,
    WORSE than flat 5.00 BPW bit-packed. Not implemented.
  - Implemented instead:
    (1) tile-local STC: 5b offset + 16b BF16 (21b/outlier, byte-packed per tile)
    (2) flat 5-bit bitpack: 32*5b = 20B/tile (vs 32B uint8 in RAM)

.cdq file layout (little-endian):
  header: magic[4]=b'CDQ1', u16 version=1, u16 n_tensors,
          bf16 lattice[26] (52B)
  per tensor: u16 name_len, name[bytes], u32 rows, u32 cols, u32 n_tiles
  per tile: u8 scale_f8 (1B), u8 tag_count (1B),
            idx_bits[20B] (32x5b, LSB-first),
            res_blob[ceil(tag_count*21/8)B] (5b offsets + 16b values, LSB-first)

Bit-identical guarantee: unpack -> quant_indices, scales, residuals equal
prototype tensors exactly; dequant math unchanged.
"""
import struct
import torch

MAGIC = b"CDQ1"
VERSION = 1   # 26-state, 5-bit (20B idx/tile)
VERSION2 = 2  # 16-state, 4-bit nibble (16B idx/tile)
TILE = 32
IDX_BYTES_PER_TILE = 20  # v1: 32*5/8
NIBBLE_BYTES_PER_TILE = 16  # v2: 32*4/8

CAMKII_LATTICE = torch.tensor([
    -1.0000, -0.7321, -0.5284, -0.3752, -0.2612, -0.1774, -0.1167, -0.0734,
    -0.0432, -0.0229, -0.0101, -0.0031, -0.0005, 0.0005, 0.0031, 0.0101,
    0.0229, 0.0432, 0.0734, 0.1167, 0.1774, 0.2612, 0.3752, 0.5284,
    0.7321, 1.0000,
], dtype=torch.bfloat16)

# Zero-clamped 15-state symmetric lattice (q15 = dup max). See cdq_lrr_qwen05.py.
CAMKII_LATTICE_16 = torch.tensor([
    -0.9685, -0.7525, -0.5834, -0.4414, -0.3174, -0.2048, -0.0998,
     0.0000,
     0.0998,  0.2048,  0.3174,  0.4414,  0.5834,  0.7525,  0.9685,
     0.9685,
], dtype=torch.bfloat16)


# ---------- bit utilities (LSB-first streams) ----------
def pack_5bit_row(q32: torch.Tensor) -> bytes:
    """q32: [32] uint8 values 0..25 -> 20 bytes LSB-first."""
    assert q32.numel() == 32
    acc = 0
    for i in range(32):
        acc |= (int(q32[i]) & 0x1F) << (5 * i)
    return acc.to_bytes(20, "little")


def unpack_5bit_row(buf: bytes) -> torch.Tensor:
    assert len(buf) == 20
    acc = int.from_bytes(buf, "little")
    out = torch.empty(32, dtype=torch.uint8)
    for i in range(32):
        out[i] = (acc >> (5 * i)) & 0x1F
    return out


def pack_indices_nibble(indices_32: torch.Tensor) -> bytes:
    """Pack 32 4-bit indices (0..15) into exactly 16 bytes.

    indices_32: Tensor of shape (32,) with uint8/long values in [0, 15].
    Even lanes -> low nibble, odd lanes -> high nibble.
    """
    assert indices_32.numel() == 32
    even = indices_32[0::2].to(torch.uint8) & 0x0F
    odd = indices_32[1::2].to(torch.uint8) & 0x0F
    packed = ((odd << 4) | even).to(torch.uint8)
    return bytes(packed.tolist())


def unpack_nibble_row(buf: bytes) -> torch.Tensor:
    """Inverse of pack_indices_nibble: 16 bytes -> 32 indices (0..15)."""
    assert len(buf) == 16
    out = torch.empty(32, dtype=torch.uint8)
    for i, b in enumerate(buf):
        out[2 * i] = b & 0x0F
        out[2 * i + 1] = (b >> 4) & 0x0F
    return out


def pack_residuals(offsets: torch.Tensor, values_bf16: torch.Tensor) -> bytes:
    """offsets: [R] 0..31 (5b each), values: [R] bf16 -> ceil(R*21/8)B LSB-first.
    Layout per residual: 5b offset followed by 16b bf16 bits."""
    r = len(offsets)
    if r == 0:
        return b""
    acc = 0
    pos = 0
    for o, v in zip(offsets.tolist(), values_bf16.view(torch.uint16).tolist()):
        acc |= (int(o) & 0x1F) << pos
        pos += 5
        acc |= (int(v) & 0xFFFF) << pos
        pos += 16
    return acc.to_bytes((pos + 7) // 8, "little")


def unpack_residuals(buf: bytes, tag_count: int):
    if tag_count == 0:
        return torch.empty(0, dtype=torch.uint8), torch.empty(0, dtype=torch.bfloat16)
    acc = int.from_bytes(buf, "little")
    offs = torch.empty(tag_count, dtype=torch.uint8)
    raw = torch.empty(tag_count, dtype=torch.uint16)
    pos = 0
    for i in range(tag_count):
        offs[i] = (acc >> pos) & 0x1F
        pos += 5
        raw[i] = (acc >> pos) & 0xFFFF
        pos += 16
    return offs, raw.view(torch.bfloat16)


# ---------- tile-local STC packing ----------
def pack_cdq_tile_stream(flat_w_32, quant_indices_32, outlier_mask_32):
    """Vectorized tile-local pack. Inputs [N,32]. Returns list of dicts
    (test-harness friendly) + packed byte blobs per tile."""
    assert flat_w_32.shape == quant_indices_32.shape == outlier_mask_32.shape
    N = flat_w_32.shape[0]
    out = []
    for i in range(N):
        mask = outlier_mask_32[i].bool()
        offs = torch.nonzero(mask).squeeze(-1).to(torch.uint8)
        vals = flat_w_32[i][mask].to(torch.bfloat16)
        out.append({
            "num_tags": int(mask.sum()),
            "local_offsets": offs,
            "exact_values": vals,
            "indices": quant_indices_32[i].to(torch.uint8),
        })
    return out


def quant_pack_from_tensors(flat_w, quant_q, saliency_mask, scales_f8_raw):
    """Build per-tile packed records + flat bitstream blobs.

    flat_w/quat_q/mask: [N,32]; scales_f8_raw: [N,1] uint8 raw FP8 bits.
    Returns dict of tensors + blobs for .cdq serialization.
    """
    from scripts.cdq_lrr_qwen05 import CAMKII_LATTICE  # noqa
    tiles = pack_cdq_tile_stream(flat_w, quant_q, saliency_mask)
    idx_blob = b"".join(pack_5bit_row(t["indices"]) for t in tiles)
    res_blobs = [pack_residuals(t["local_offsets"], t["exact_values"]) for t in tiles]
    tag_counts = torch.tensor([t["num_tags"] for t in tiles], dtype=torch.uint8)
    return {"tiles": tiles, "idx_blob": idx_blob, "res_blobs": res_blobs,
            "tag_counts": tag_counts}


# ---------- .cdq writer / reader ----------
def _bf16_to_u16(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint16).numpy().tobytes()


def write_cdq(path, tensors: dict, n_states: int = 26):
    """tensors: {name: dict(rows, cols, scales_f8_u8[N,1], q[N,32]u8,
    tag_counts[N]u8, res_blobs[list])}.

    n_states=26 -> file version 1 (20B 5-bit idx/tile).
    n_states=16 -> file version 2 (16B nibble idx/tile).
    Residual variable-length layout is identical in both versions.
    """
    assert n_states in (26, 16)
    ver = VERSION if n_states == 26 else VERSION2
    lat = CAMKII_LATTICE if n_states == 26 else CAMKII_LATTICE_16
    idx_bytes = IDX_BYTES_PER_TILE if n_states == 26 else NIBBLE_BYTES_PER_TILE
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<HH", ver, len(tensors)))
        f.write(_bf16_to_u16(lat))
        for name, t in tensors.items():
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<III", t["rows"], t["cols"], t["n_tiles"]))
            for i in range(t["n_tiles"]):
                f.write(bytes([int(t["scales_f8_u8"][i, 0])]))
                f.write(bytes([int(t["tag_counts"][i])]))
                qb = (pack_5bit_row(t["q"][i]) if n_states == 26
                      else pack_indices_nibble(t["q"][i]))
                assert len(qb) == idx_bytes
                f.write(qb)
                f.write(t["res_blobs"][i])


def read_cdq(path):
    with open(path, "rb") as f:
        buf = f.read()
    p = 0
    assert buf[p:p + 4] == MAGIC, "bad magic"
    p += 4
    ver, n = struct.unpack_from("<HH", buf, p)
    p += 4
    assert ver in (VERSION, VERSION2), f"unsupported version {ver}"
    n_states = 26 if ver == VERSION else 16
    p += 52 if ver == VERSION else 32  # lattice
    idx_bytes = IDX_BYTES_PER_TILE if ver == VERSION else NIBBLE_BYTES_PER_TILE
    unpack_row = unpack_5bit_row if ver == VERSION else unpack_nibble_row
    out = {}
    for _ in range(n):
        (nl,) = struct.unpack_from("<H", buf, p)
        p += 2
        name = buf[p:p + nl].decode()
        p += nl
        rows, cols, nt = struct.unpack_from("<III", buf, p)
        p += 12
        scales = torch.empty((nt, 1), dtype=torch.uint8)
        q = torch.empty((nt, 32), dtype=torch.uint8)
        tags = torch.empty(nt, dtype=torch.uint8)
        res = []
        for i in range(nt):
            scales[i, 0] = buf[p]
            p += 1
            tc = buf[p]
            p += 1
            tags[i] = tc
            q[i] = unpack_row(buf[p:p + idx_bytes])
            p += idx_bytes
            nb = (tc * 21 + 7) // 8
            rb = buf[p:p + nb]
            p += nb
            res.append(rb)
        out[name] = {"rows": rows, "cols": cols, "n_tiles": nt,
                     "scales_f8_u8": scales, "q": q,
                     "tag_counts": tags, "res_blobs": res,
                     "n_states": n_states}
    return out


def dequant_tile(q32, scale_bf16, res_blob, tag_count, lattice=None):
    """Reference tile dequant (mirrors C++/Metal kernels)."""
    lat = CAMKII_LATTICE if lattice is None else lattice
    w = lat[q32.long()].to(torch.bfloat16) * scale_bf16
    if tag_count:
        offs, vals = unpack_residuals(res_blob, int(tag_count))
        # residuals are bit-exact originals; locate via local offset
        for o, v in zip(offs.tolist(), vals.tolist() if vals.dtype != torch.bfloat16 else vals):
            pass
        offs_t, vals_t = unpack_residuals(res_blob, int(tag_count))
        w[offs_t.long()] = vals_t.to(torch.bfloat16)
    return w


def main():
    import argparse, time, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("CDQ_MODEL", "models/qwen-local"))
    ap.add_argument("--out", default="qwen05.cdq")
    ap.add_argument("--max-tensors", type=int, default=-1, help="-1 = all 168")
    ap.add_argument("--states", type=int, choices=(26, 16), default=16,
                    help="lattice states: 26 (v1 5-bit) or 16 (v2 4-bit nibble)")
    ap.add_argument("--stc-frac", type=float, default=0.0156,
                    help="STC outlier fraction (0.0156 = top 1.56%, 0.03125 = 1/tile)")
    args = ap.parse_args()

    from safetensors import safe_open
    lattice = CAMKII_LATTICE if args.states == 26 else CAMKII_LATTICE_16
    index_bits = 5 if args.states == 26 else 4
    print(f"lattice: {args.states}-state ({index_bits}-bit indices)")
    f = safe_open(os.path.join(args.model, "model.safetensors"), framework="pt")
    keys = [k for k in f.keys()
            if k.endswith(".weight")
            and any(s in k for s in ("q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"))]
    if args.max_tensors > 0:
        keys = keys[:args.max_tensors]
    print(f"packing {len(keys)} tensors ...")
    lattice_f = lattice.float()
    tensors, tot_w, tot_tiles, tot_res = {}, 0, 0, 0
    t0 = time.time()
    for ki, k in enumerate(keys):
        w = f.get_tensor(k).to(torch.bfloat16)
        rows, cols = w.shape
        flat = w.view(-1, 32)
        n = flat.shape[0]
        sigma = flat.abs().float().amax(dim=1, keepdim=True).clamp_min(1e-8)
        sigma_q = sigma.to(torch.float8_e4m3fn).to(torch.bfloat16)
        f8raw = sigma.to(torch.float8_e4m3fn).view(torch.uint8).reshape(n, 1)
        var = flat.float().var(dim=1, keepdim=True)
        sal = flat.float().abs() * (var + 1e-6)
        thr = torch.quantile(sal.flatten(), 1.0 - args.stc_frac)
        mask = sal >= thr
        norm = flat.float() / sigma_q.float()
        q = torch.argmin((norm.unsqueeze(-1) - lattice_f).abs(), dim=-1).to(torch.uint8)
        # tile-local residuals
        res_blobs, tags = [], torch.empty(n, dtype=torch.uint8)
        for i in range(n):
            m = mask[i]
            offs = torch.nonzero(m).squeeze(-1).to(torch.uint8)
            vals = flat[i][m].to(torch.bfloat16)
            tags[i] = int(m.sum())
            res_blobs.append(pack_residuals(offs, vals))
        tensors[k] = {"rows": rows, "cols": cols, "n_tiles": n,
                      "scales_f8_u8": f8raw, "q": q,
                      "tag_counts": tags, "res_blobs": res_blobs}
        tot_w += flat.numel()
        tot_tiles += n
        tot_res += int(mask.sum())
        if (ki + 1) % 40 == 0:
            print(f"  [{ki+1}/{len(keys)}] {k}")
    write_cdq(args.out, tensors, n_states=args.states)
    dt = time.time() - t0
    size = os.path.getsize(args.out)
    # accounting: header + per-tile fixed + residuals
    bpw = size * 8 / tot_w
    print(f"wrote {args.out} ({size/1e6:.1f} MB) in {dt:.1f}s")
    print(f"params={tot_w} tiles={tot_tiles} residuals={tot_res} "
          f"({100*tot_res/tot_w:.2f}%)")
    print(f"on-disk = {bpw:.3f} bpw")
    print(f"budget: {index_bits:.3f} idx + 0.250 scale + "
          f"{tot_res*21/tot_w:.3f} res(21b) + header -> ~{(tot_tiles*16+size*0)/tot_w:.3f}+hdr")


if __name__ == "__main__":
    main()
