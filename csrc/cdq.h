// CDQ-LRR tile dequantizer — portable C++17 reference.
// Mirrors scripts/cdq_pack.py layout (LSB-first bit streams).
// Tile: 1B FP8 scale (E4M3) + 1B tag_count + 20B idx (32x5b) + res_blob.
//
// FP8 E4M3 decode: s.eeee.mmm (bias 8, max 448, NaN 0x7F/0xFF reserved).
// Outputตรง half/BF16 left to caller; here we decode to float.
#pragma once
#include <cstdint>
#include <cstring>

namespace cdq {

// 26-state CaMKII lattice (float copy of BF16 centroids, v1).
static const float LATTICE[26] = {
    -1.0000f, -0.7321f, -0.5284f, -0.3752f, -0.2612f, -0.1774f, -0.1167f, -0.0734f,
    -0.0432f, -0.0229f, -0.0101f, -0.0031f, -0.0005f,  0.0005f,  0.0031f,  0.0101f,
     0.0229f,  0.0432f,  0.0734f,  0.1167f,  0.1774f,  0.2612f,  0.3752f,  0.5284f,
     0.7321f,  1.0000f,
};

// Zero-clamped 15-state symmetric lattice in 16 slots (q15 = dup max, v2).
static const float LATTICE16[16] = {
    -0.9685f, -0.7525f, -0.5834f, -0.4414f, -0.3174f, -0.2048f, -0.0998f,
     0.0000f,
     0.0998f,  0.2048f,  0.3174f,  0.4414f,  0.5834f,  0.7525f,  0.9685f,
     0.9685f,
};

inline float fp8_e4m3_to_f32(uint8_t b) {
    uint32_t sign = (b >> 7) & 1;
    uint32_t exp  = (b >> 3) & 0xF;
    uint32_t man  = b & 0x7;
    uint32_t f;
    if (exp == 0) {
        // subnormal: 2^-8 * (man/8)  (E4M3 bias 8; min normal 2^-6... use ldexp path)
        float m = (float)man / 8.0f;
        float v = m * 0.00390625f;  // 2^-8
        return sign ? -v : v;
    }
    if (exp == 0xF) {  // inf/nan -> saturate to 448 (match torch conversion clamp)
        return sign ? -448.0f : 448.0f;
    }
    // normal: 2^(exp-8) * (1 + man/8)
    int e = (int)exp - 8;
    float m = 1.0f + (float)man / 8.0f;
    float v = m;
    while (e > 0) { v *= 2.0f; e--; }
    while (e < 0) { v *= 0.5f; e++; }
    return sign ? -v : v;
}

// Unpack 20B LSB-first -> 32 indices (0..25).
inline void unpack_idx5(const uint8_t* in20, uint8_t* out32) {
    // generic bit reader (simple, correct; 32*5 bits = 160 bits)
    int bit = 0;
    for (int i = 0; i < 32; i++) {
        int byte = bit >> 3, shift = bit & 7;
        uint32_t w = (uint32_t)in20[byte] | ((uint32_t)(byte + 1 < 20 ? in20[byte + 1] : 0) << 8) |
                     ((uint32_t)(byte + 2 < 20 ? in20[byte + 2] : 0) << 16);
        out32[i] = (w >> shift) & 0x1F;
        bit += 5;
    }
}

// Unpack 16B nibbles -> 32 indices (0..15). Even lane = low nibble.
inline void unpack_nibble(const uint8_t* in16, uint8_t* out32) {
    for (int i = 0; i < 16; i++) {
        out32[2 * i] = in16[i] & 0x0F;
        out32[2 * i + 1] = (in16[i] >> 4) & 0x0F;
    }
}
// Residual blob: per residual 5b offset + 16b BF16 bits, LSB-first.
// bf16_out: caller buffer for R uint16 raw bits; off_out: R offsets.
inline void unpack_res(const uint8_t* blob, int blob_bytes, int tag_count,
                       uint8_t* off_out, uint16_t* bf16_out) {
    int bit = 0;
    int total_bits = blob_bytes * 8;
    for (int r = 0; r < tag_count; r++) {
        uint32_t acc = 0;
        for (int b = 0; b < 4 && (bit >> 3) + b < total_bits / 8 + 1; b++) {
            int bi = (bit >> 3) + b;
            if (bi < blob_bytes) acc |= (uint32_t)blob[bi] << (8 * b);
        }
        int shift = bit & 7;
        off_out[r] = (acc >> shift) & 0x1F;
        bit += 5;
        acc = 0;
        for (int b = 0; b < 4; b++) {
            int bi = (bit >> 3) + b;
            if (bi < blob_bytes) acc |= (uint32_t)blob[bi] << (8 * b);
        }
        shift = bit & 7;
        bf16_out[r] = (acc >> shift) & 0xFFFF;
        bit += 16;
    }
}

inline uint16_t f32_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    return (uint16_t)(u >> 16);  // truncation matches bit-exact needs only for residuals path
}

// Dequantize one tile -> 32 floats. res_blob may be nullptr when tag_count==0.
inline void dequant_tile(uint8_t scale_f8, const uint8_t idx20[20],
                         const uint8_t* res_blob, int res_bytes, int tag_count,
                         float out32[32]) {
    float sigma = fp8_e4m3_to_f32(scale_f8);
    uint8_t q[32];
    unpack_idx5(idx20, q);
    for (int i = 0; i < 32; i++) out32[i] = sigma * LATTICE[q[i] & 0x1F];
    if (tag_count > 0) {
        uint8_t off[64];
        uint16_t raw[64];
        int R = tag_count > 64 ? 64 : tag_count;
        unpack_res(res_blob, res_bytes, R, off, raw);
        for (int r = 0; r < R; r++) {
            uint8_t o = off[r] & 0x1F;
            // BF16 bits -> float
            uint32_t u = (uint32_t)raw[r] << 16;
            float v;
            memcpy(&v, &u, 4);
            out32[o] = v;
        }
    }
}

// v2: dequantize one 4-bit nibble tile -> 32 floats.
inline void dequant_tile_4bit(uint8_t scale_f8, const uint8_t idx16[16],
                              const uint8_t* res_blob, int res_bytes, int tag_count,
                              float out32[32]) {
    float sigma = fp8_e4m3_to_f32(scale_f8);
    uint8_t q[32];
    unpack_nibble(idx16, q);
    for (int i = 0; i < 32; i++) out32[i] = sigma * LATTICE16[q[i] & 0x0F];
    if (tag_count > 0) {
        uint8_t off[64];
        uint16_t raw[64];
        int R = tag_count > 64 ? 64 : tag_count;
        unpack_res(res_blob, res_bytes, R, off, raw);
        for (int r = 0; r < R; r++) {
            uint32_t u = (uint32_t)raw[r] << 16;
            float v;
            memcpy(&v, &u, 4);
            out32[off[r] & 0x1F] = v;
        }
    }
}

}  // namespace cdq
