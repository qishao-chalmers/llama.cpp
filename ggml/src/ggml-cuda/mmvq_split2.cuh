// mmvq_split2.cuh — single-buffer split (A+B) Q8_0 GEMV kernels
#pragma once

#include "common.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define QK8_0_SPLIT2 32
#define QI8_0_SPLIT2 4   // ra/rb are 16B each -> 4 int32

// Draft consumes one int32 from ra per vec_dot call.
#define VDR_Q8_0_SPLIT2_DRAFT_Q8_1_MMVQ 1
// Verify reconstructs full int8 and consumes two int32 packs (like Q8_0).
#define VDR_Q8_0_SPLIT2_Q8_1_MMVQ 2

static_assert(sizeof(block_q8_0_split2) == 2*sizeof(ggml_half) + QK8_0_SPLIT2, "block_q8_0_split2 size mismatch");

static __device__ __forceinline__ int8_t split2_nib_sign_ext(int n) {
    return (int8_t)((n ^ 8) - 8);
}

static __device__ __forceinline__ int split2_nib_unpack4(uint8_t lo, uint8_t hi) {
    const int8_t n0 = split2_nib_sign_ext(lo >> 4);
    const int8_t n1 = split2_nib_sign_ext(lo & 0xF);
    const int8_t n2 = split2_nib_sign_ext(hi >> 4);
    const int8_t n3 = split2_nib_sign_ext(hi & 0xF);
    return (uint8_t)n0 | ((uint8_t)n1 << 8) | ((uint8_t)n2 << 16) | ((uint8_t)n3 << 24);
}

// Draft: identical math as q8_0_nib, but reads d_draft from split2.
static __device__ __forceinline__ float vec_dot_q8_0_split2_draft_q8_1(
    const void         * __restrict__ vbq,
    const block_q8_1   * __restrict__ bq8_1,
    const int          & kbx,
    const int          & iqs)
{
    const block_q8_0_split2 * bq = (const block_q8_0_split2 *) vbq + kbx;

    // ra is 4B-aligned for split2 (sizeof(block)=36), so b4 load is safe.
    const int ra_i32 = get_int_b4(bq->ra, iqs);

    const int v0 = split2_nib_unpack4((ra_i32 >>  0) & 0xFF, (ra_i32 >>  8) & 0xFF);
    const int v1 = split2_nib_unpack4((ra_i32 >> 16) & 0xFF, (ra_i32 >> 24) & 0xFF);

    const int u0 = get_int_b4(bq8_1->qs, iqs * 2);
    const int u1 = get_int_b4(bq8_1->qs, iqs * 2 + 1);

    const int sumi = ggml_cuda_dp4a(v0, u0, ggml_cuda_dp4a(v1, u1, 0));
    return __half2float(bq->d_draft) * __low2float(bq8_1->ds) * (float)sumi;
}

// Verify: reconstruct full int8 q = (hi<<4)|lo using ra/rb, then reuse Q8_0 dot core.
static __device__ __forceinline__ float vec_dot_q8_0_split2_q8_1(
    const void         * __restrict__ vbq,
    const block_q8_1   * __restrict__ bq8_1,
    const int          & kbx,
    const int          & iqs)
{
    const block_q8_0_split2 * bq = (const block_q8_0_split2 *) vbq + kbx;

    const int ra_i32 = get_int_b4(bq->ra, iqs);
    const int rb_i32 = get_int_b4(bq->rb, iqs);

    // Reconstruct 8 int8 values for this iqs (covers 8 elements):
    // Each int32 holds 4 packed bytes, each byte holds two nibbles.
    // For each nibble-pair we create one signed int8: (hi<<4)|lo.
    const uint8_t a0 = (ra_i32 >>  0) & 0xFF; const uint8_t b0 = (rb_i32 >>  0) & 0xFF;
    const uint8_t a1 = (ra_i32 >>  8) & 0xFF; const uint8_t b1 = (rb_i32 >>  8) & 0xFF;
    const uint8_t a2 = (ra_i32 >> 16) & 0xFF; const uint8_t b2 = (rb_i32 >> 16) & 0xFF;
    const uint8_t a3 = (ra_i32 >> 24) & 0xFF; const uint8_t b3 = (rb_i32 >> 24) & 0xFF;

    const int8_t q0 = (int8_t)(((a0 >> 4) << 4) | (b0 >> 4));
    const int8_t q1 = (int8_t)(((a0 & 0x0F) << 4) | (b0 & 0x0F));
    const int8_t q2 = (int8_t)(((a1 >> 4) << 4) | (b1 >> 4));
    const int8_t q3 = (int8_t)(((a1 & 0x0F) << 4) | (b1 & 0x0F));
    const int8_t q4 = (int8_t)(((a2 >> 4) << 4) | (b2 >> 4));
    const int8_t q5 = (int8_t)(((a2 & 0x0F) << 4) | (b2 & 0x0F));
    const int8_t q6 = (int8_t)(((a3 >> 4) << 4) | (b3 >> 4));
    const int8_t q7 = (int8_t)(((a3 & 0x0F) << 4) | (b3 & 0x0F));

    const int v0 = (uint8_t)q0 | ((uint8_t)q1 << 8) | ((uint8_t)q2 << 16) | ((uint8_t)q3 << 24);
    const int v1 = (uint8_t)q4 | ((uint8_t)q5 << 8) | ((uint8_t)q6 << 16) | ((uint8_t)q7 << 24);

    const int u0 = get_int_b4(bq8_1->qs, iqs * 2);
    const int u1 = get_int_b4(bq8_1->qs, iqs * 2 + 1);

    const int sumi = ggml_cuda_dp4a(v0, u0, ggml_cuda_dp4a(v1, u1, 0));
    return __half2float(bq->d_full) * __low2float(bq8_1->ds) * (float)sumi;
}

