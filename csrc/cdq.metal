// CDQ-LRR kernels — Apple Silicon (Metal Shading Language).
//
// .cdq v1 (26-state): 1B FP8 scale + 1B tag_count + 20B idx (32x5b, LSB-first)
// .cdq v2 (16-state): 1B FP8 scale + 1B tag_count + 16B idx (32x4b nibbles)
// Residual blob (both): 5b tile-local offset + 16b BF16 bits per outlier, LSB-first.
//
// v2 nibble layout: even lane -> low nibble, odd lane -> high nibble, so
// lane unpack is two ALU ops with no cross-byte shifts:
//   q = (idx[lane >> 1] >> ((lane & 1) << 2)) & 0xF
#include <metal_stdlib>
using namespace metal;

// 26-state lattice (v1, float for register gather).
constant float LATTICE26[26] = {
    -1.0000, -0.7321, -0.5284, -0.3752, -0.2612, -0.1774, -0.1167, -0.0734,
    -0.0432, -0.0229, -0.0101, -0.0031, -0.0005,  0.0005,  0.0031,  0.0101,
     0.0229,  0.0432,  0.0734,  0.1167,  0.1774,  0.2612,  0.3752,  0.5284,
     0.7321,  1.0000,
};

// Zero-clamped 15-state symmetric lattice in 16 slots (v2, q15 = dup max).
constant float LATTICE16[16] = {
    -0.9685, -0.7525, -0.5834, -0.4414, -0.3174, -0.2048, -0.0998,
     0.0000,
     0.0998,  0.2048,  0.3174,  0.4414,  0.5834,  0.7525,  0.9685,
     0.9685,
};

inline float fp8_e4m3_to_float(uchar b) {
    uint s = (b >> 7) & 1;
    uint e = (b >> 3) & 0xF;
    uint m = b & 0x7;
    if (e == 0) {
        float v = (float)m / 8.0 * 0.00390625; // 2^-8 subnormal
        return s ? -v : v;
    }
    if (e == 0xF) return s ? -448.0 : 448.0; // saturate NaN/Inf like torch
    float v = ldexp(1.0 + (float)m / 8.0, (int)e - 8);
    return s ? -v : v;
}

inline float bf16_to_float(ushort raw) {
    return as_type<float>(uint(raw) << 16);
}

// ---- v1: 5-bit tile dequant (one thread per tile) ----
kernel void cdq_dequant_tile(
    device const uchar*  scales     [[buffer(0)]],
    device const uchar*  tags       [[buffer(1)]],
    device const uchar*  idx20      [[buffer(2)]],
    device const uchar*  res_blob   [[buffer(3)]],
    device const uint*   res_base   [[buffer(4)]],
    device half*         out        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    float sigma = fp8_e4m3_to_float(scales[tid]);
    device const uchar* ip = idx20 + tid * 20;
    float w[32];
    int bit = 0;
    for (int i = 0; i < 32; i++) {
        int by = bit >> 3, sh = bit & 7;
        uint word = (uint)ip[by] | ((uint)(by + 1 < 20 ? ip[by + 1] : 0) << 8)
                  | ((uint)(by + 2 < 20 ? ip[by + 2] : 0) << 16);
        w[i] = sigma * LATTICE26[(word >> sh) & 0x1F];
        bit += 5;
    }
    uint tc = tags[tid];
    if (tc > 0) {
        device const uchar* rp = res_blob + res_base[tid];
        int rbit = 0;
        for (uint r = 0; r < tc; r++) {
            int by = rbit >> 3, sh = rbit & 7;
            uint acc = (uint)rp[by] | ((uint)rp[by+1] << 8) | ((uint)rp[by+2] << 16);
            uint off = (acc >> sh) & 0x1F;
            rbit += 5;
            by = rbit >> 3; sh = rbit & 7;
            acc = (uint)rp[by] | ((uint)rp[by+1] << 8) | ((uint)rp[by+2] << 16);
            w[off & 31] = bf16_to_float((acc >> sh) & 0xFFFF);
            rbit += 16;
        }
    }
    device half* op = out + tid * 32;
    for (int i = 0; i < 32; i++) op[i] = half(w[i]);
}

// ---- v2: 4-bit nibble tile dequant (one thread per tile) ----
kernel void cdq_dequant_tile_4bit(
    device const uchar*  scales     [[buffer(0)]],
    device const uchar*  tags       [[buffer(1)]],
    device const uchar*  idx16      [[buffer(2)]],
    device const uchar*  res_blob   [[buffer(3)]],
    device const uint*   res_base   [[buffer(4)]],
    device half*         out        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    float sigma = fp8_e4m3_to_float(scales[tid]);
    device const uchar* ip = idx16 + tid * 16;
    float w[32];
    for (int lane = 0; lane < 32; lane++)
        w[lane] = sigma * LATTICE16[(ip[lane >> 1] >> ((lane & 1) << 2)) & 0xF];
    uint tc = tags[tid];
    if (tc > 0) {
        device const uchar* rp = res_blob + res_base[tid];
        int rbit = 0;
        for (uint r = 0; r < tc; r++) {
            int by = rbit >> 3, sh = rbit & 7;
            uint acc = (uint)rp[by] | ((uint)rp[by+1] << 8) | ((uint)rp[by+2] << 16);
            uint off = (acc >> sh) & 0x1F;
            rbit += 5;
            by = rbit >> 3; sh = rbit & 7;
            acc = (uint)rp[by] | ((uint)rp[by+1] << 8) | ((uint)rp[by+2] << 16);
            w[off & 31] = bf16_to_float((acc >> sh) & 0xFFFF);
            rbit += 16;
        }
    }
    device half* op = out + tid * 32;
    for (int i = 0; i < 32; i++) op[i] = half(w[i]);
}

// ---- v2: fused GEMV, 1 SIMDgroup (32 lanes) cooperates per output row ----
// Grid: threadgroup_pos.x = row block; 4 SIMDgroups per threadgroup -> 4 rows.
// K_tiles = number of 32-tiles per row. activations: K_tiles*32 halves.
kernel void cdq_lrr_gemv_4bit(
    device const uchar*  scales     [[buffer(0)]],
    device const uchar*  tags       [[buffer(1)]],
    device const uchar*  idx16      [[buffer(2)]],
    device const uchar*  res_blob   [[buffer(3)]],
    device const uint*   res_base   [[buffer(4)]],
    device const half*   activations [[buffer(5)]],
    device half*         outputs    [[buffer(6)]],
    constant uint&       K_tiles    [[buffer(7)]],
    uint2                threadgroup_pos [[threadgroup_position_in_grid]],
    uint                 simd_lane_id    [[thread_index_in_simdgroup]],
    uint                 simd_group_id   [[simdgroup_index_in_threadgroup]])
{
    uint row = threadgroup_pos.x * 4 + simd_group_id;
    uint lane = simd_lane_id; // 0..31
    float acc = 0.0;
    uint byte_idx = lane >> 1;
    uint shift = (lane & 1) << 2;
    for (uint t = 0; t < K_tiles; ++t) {
        uint ti = row * K_tiles + t;
        float sigma = fp8_e4m3_to_float(scales[ti]);
        device const uchar* ip = idx16 + ti * 16;
        float w = sigma * LATTICE16[(ip[byte_idx] >> shift) & 0xF];
        uint tc = tags[ti];
        if (tc > 0) {
            // avg 0.5 tags/tile: linear scan over <= few residuals is cheap
            device const uchar* rp = res_blob + res_base[ti];
            int rbit = 0;
            for (uint r = 0; r < tc; r++) {
                int by = rbit >> 3, sh = rbit & 7;
                uint aw = (uint)rp[by] | ((uint)rp[by+1] << 8) | ((uint)rp[by+2] << 16);
                uint off = (aw >> sh) & 0x1F;
                rbit += 5;
                by = rbit >> 3; sh = rbit & 7;
                aw = (uint)rp[by] | ((uint)rp[by+1] << 8) | ((uint)rp[by+2] << 16);
                if (off == lane) w = bf16_to_float((aw >> sh) & 0xFFFF);
                rbit += 16;
            }
        }
        acc += w * float(activations[t * 32 + lane]);
    }
    float row_sum = simd_sum(acc);
    if (lane == 0) outputs[row] = half(row_sum);
}
