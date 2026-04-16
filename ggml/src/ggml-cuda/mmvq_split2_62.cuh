// mmvq_split2_62.cuh — Q8_0 split with A=6 MSBs, B=2 LSBs per int8
#pragma once

#include "common.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define VDR_Q8_0_SPLIT2_62_DRAFT_Q8_1_MMVQ 1
#define VDR_Q8_0_SPLIT2_62_Q8_1_MMVQ       1

static_assert(sizeof(block_q8_0_split2_62) == 2 * sizeof(ggml_half) + QK8_0, "block_q8_0_split2_62 size mismatch");

static __device__ __forceinline__ int split2_62_sign_ext6(unsigned h) {
    h &= 0x3F;
    if (h & 0x20) {
        return (int) h - 64;
    }
    return (int) h;
}

// Unpack 8×6-bit values from 6 consecutive bytes (48 bits).
static __device__ __forceinline__ void split2_62_unpack6_8(const uint8_t * GGML_RESTRICT ra6, uint8_t * GGML_RESTRICT hi) {
    uint64_t bit = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t v = 0;
#pragma unroll
        for (int b = 0; b < 6; ++b) {
            if (ra6[bit / 8] & (1u << (bit % 8))) {
                v |= (1u << b);
            }
            bit++;
        }
        hi[i] = (uint8_t)(v & 0x3F);
    }
}

static __device__ __forceinline__ void split2_62_unpack2_8(const uint8_t * GGML_RESTRICT rb2, uint8_t * GGML_RESTRICT lo) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        lo[i] = (rb2[i / 4] >> ((i % 4) * 2)) & 3;
    }
}

static __device__ __forceinline__ float vec_dot_q8_0_split2_62_draft_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int & kbx,
    const int & iqs)
{
    const block_q8_0_split2_62 * bq = (const block_q8_0_split2_62 *) vbq + kbx;
    const uint8_t * ra = bq->ra + iqs * 6;
    uint8_t hi[8];
    split2_62_unpack6_8(ra, hi);
    int8_t qv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        qv[i] = (int8_t) split2_62_sign_ext6(hi[i]);
    }
    const int v0 = (uint8_t) qv[0] | ((uint8_t) qv[1] << 8) | ((uint8_t) qv[2] << 16) | ((uint8_t) qv[3] << 24);
    const int v1 = (uint8_t) qv[4] | ((uint8_t) qv[5] << 8) | ((uint8_t) qv[6] << 16) | ((uint8_t) qv[7] << 24);

    const int u0 = get_int_b4(bq8_1->qs, iqs * 2);
    const int u1 = get_int_b4(bq8_1->qs, iqs * 2 + 1);

    const int sumi = ggml_cuda_dp4a(v0, u0, ggml_cuda_dp4a(v1, u1, 0));
    return __half2float(bq->d_draft) * __low2float(bq8_1->ds) * (float) sumi;
}

static __device__ __forceinline__ float vec_dot_q8_0_split2_62_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int & kbx,
    const int & iqs)
{
    const block_q8_0_split2_62 * bq = (const block_q8_0_split2_62 *) vbq + kbx;
    const uint8_t * ra = bq->ra + iqs * 6;
    const uint8_t * rb = bq->rb + iqs * 2;
    uint8_t hi[8];
    uint8_t lo[8];
    split2_62_unpack6_8(ra, hi);
    split2_62_unpack2_8(rb, lo);
    int8_t qv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint8_t u = (uint8_t)(((hi[i] & 0x3F) << 2) | (lo[i] & 3));
        qv[i] = (int8_t) u;
    }
    const int v0 = (uint8_t) qv[0] | ((uint8_t) qv[1] << 8) | ((uint8_t) qv[2] << 16) | ((uint8_t) qv[3] << 24);
    const int v1 = (uint8_t) qv[4] | ((uint8_t) qv[5] << 8) | ((uint8_t) qv[6] << 16) | ((uint8_t) qv[7] << 24);

    const int u0 = get_int_b4(bq8_1->qs, iqs * 2);
    const int u1 = get_int_b4(bq8_1->qs, iqs * 2 + 1);

    const int sumi = ggml_cuda_dp4a(v0, u0, ggml_cuda_dp4a(v1, u1, 0));
    return __half2float(bq->d_full) * __low2float(bq8_1->ds) * (float) sumi;
}
