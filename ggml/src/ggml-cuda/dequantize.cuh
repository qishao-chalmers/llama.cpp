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
