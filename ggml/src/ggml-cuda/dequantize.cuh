#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// block_q8_0_split2: full dequant (reconstruct int8 from upper/lower nibbles, scale by d_full)
static __device__ __forceinline__ void dequantize_q8_0_split2(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q8_0_split2 * x = (const block_q8_0_split2 *) vx;

    const float d = x[ib].d_full;
    const int j = iqs / 2;

    const uint8_t a = x[ib].ra[j];
    const uint8_t b = x[ib].rb[j];

    const int hi0 = (int) (a >> 4);
    const int lo0 = (int) (b >> 4);
    const int hi1 = (int) (a & 0x0F);
    const int lo1 = (int) (b & 0x0F);

    const int q0 = (hi0 << 4) | lo0;
    const int q1 = (hi1 << 4) | lo1;

    v.x = (float) ((int8_t) q0) * d;
    v.y = (float) ((int8_t) q1) * d;
}

// block_q8_0_split2: draft path (upper nibbles only, d_draft scale)
static __device__ __forceinline__ void dequantize_q8_0_split2_draft(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q8_0_split2 * x = (const block_q8_0_split2 *) vx;

    const float d = x[ib].d_draft;
    const int j = iqs / 2;

    const uint8_t packed = x[ib].ra[j];
    const int n0u = packed >> 4;
    const int n1u = packed & 0x0F;
    const int n0 = (n0u ^ 8) - 8;
    const int n1 = (n1u ^ 8) - 8;

    v.x = (float) n0 * d;
    v.y = (float) n1 * d;
}

// block_q8_0_split2_62: bit layout matches ggml-quants.c (LSB-first 6-bit stream for ra; 2 bits/weight in rb).
static __device__ __forceinline__ uint8_t q8_0_split2_62_get_hi6(const uint8_t * ra, int idx) {
    uint64_t bit = (uint64_t) idx * 6;
    uint32_t v = 0;
#pragma unroll
    for (int b = 0; b < 6; ++b) {
        if (ra[bit / 8] & (1u << (bit % 8))) {
            v |= (1u << b);
        }
        bit++;
    }
    return (uint8_t)(v & 0x3F);
}

static __device__ __forceinline__ uint8_t q8_0_split2_62_get_lo2(const uint8_t * rb, int idx) {
    return (rb[idx / 4] >> ((idx % 4) * 2)) & 3;
}

static __device__ __forceinline__ int q8_0_sign_ext6_dev(uint32_t h) {
    h &= 0x3F;
    if (h & 0x20) {
        return (int) h - 64;
    }
    return (int) h;
}

static __device__ __forceinline__ void dequantize_q8_0_split2_62(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q8_0_split2_62 * x = (const block_q8_0_split2_62 *) vx;

    const float d = x[ib].d_full;
    const int i0 = iqs;
    const int i1 = iqs + 1;

    const uint8_t h0 = q8_0_split2_62_get_hi6(x[ib].ra, i0);
    const uint8_t h1 = q8_0_split2_62_get_hi6(x[ib].ra, i1);
    const uint8_t lo0 = q8_0_split2_62_get_lo2(x[ib].rb, i0);
    const uint8_t lo1 = q8_0_split2_62_get_lo2(x[ib].rb, i1);

    const int q0 = (int) (((h0 & 0x3F) << 2) | (lo0 & 3));
    const int q1 = (int) (((h1 & 0x3F) << 2) | (lo1 & 3));

    v.x = (float) ((int8_t) (uint8_t) q0) * d;
    v.y = (float) ((int8_t) (uint8_t) q1) * d;
}

static __device__ __forceinline__ void dequantize_q8_0_split2_62_draft(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q8_0_split2_62 * x = (const block_q8_0_split2_62 *) vx;

    const float d = x[ib].d_draft;
    const int i0 = iqs;
    const int i1 = iqs + 1;

    const uint8_t h0 = q8_0_split2_62_get_hi6(x[ib].ra, i0);
    const uint8_t h1 = q8_0_split2_62_get_hi6(x[ib].ra, i1);

    v.x = (float) q8_0_sign_ext6_dev(h0) * d;
    v.y = (float) q8_0_sign_ext6_dev(h1) * d;
}
