// mmvq_nib.cuh — Nibble-draft Q8_0 GEMV kernel for split-weight precision
//
// Stores only the upper-4-bit nibble of each Q8_0 int8 weight, halving the
// weight bytes loaded per GEMV.  With a properly HBM-bound kernel this yields
// ~1.8× wall-clock speedup vs the full Q8_0 kernel (53.5→28.3 MB/layer).
//
// Block layout — block_q8_0_nib (18 bytes, vs 34 for block_q8_0):
//
//   offset  0: ggml_half d          — draft_scale = block_scale × 16
//   offset  2: uint8_t  ra[16]      — upper nibbles packed 2-per-byte
//               ra[j] = (upper_nibble(qs[2j]) << 4) | upper_nibble(qs[2j+1])
//               upper_nibble(int8 q) = ((uint8_t)q) >> 4
//
// Dequant: value_i = sign_ext4(nibble_i) × draft_scale
//           sign_ext4: (n ^ 8) - 8  maps unsigned nibble [0..15] → [-8..7]
//
// vec_dot: iqs ∈ {0,1,2,3} selects one int32 of ra (4 bytes = 8 nibbles).
//   QI  = 4  (int32s per nibble block ra buffer)
//   VDR = 1  (int32s consumed per vec_dot call)
//   kbx = tid / (QI/VDR) = tid/4   — same block assignment as Q8_0 (tid/4)
//   blocks_per_iter = VDR × nwarps × warp_size / QI = 1×4×32/4 = 32   (same)
//   → outer loop executes the same number of iterations as Q8_0, loading half
//     the bytes.  All compute overhead (nibble unpack + dp4a) hides behind
//     memory latency; the kernel remains HBM-bound at scale.

#pragma once

#include "common.cuh"
#include "vecdotq.cuh"

#include <cstdint>

// ── Constants ────────────────────────────────────────────────────────────────
#define QK8_0_NIB 32    // elements per nibble block (matches QK8_0)
#define QI8_0_NIB  4    // int32s in ra:  16 bytes / 4 bytes-per-int32 = 4
// VDR for the MMVQ path: one int32 of ra per thread per vec_dot call
// (handles 8 elements via two dp4a calls)
#define VDR_Q8_0_NIB_Q8_1_MMVQ 1

// ── Block struct ─────────────────────────────────────────────────────────────
// Host/device type lives in ggml-common.h as block_q8_0_nib.
static_assert(sizeof(block_q8_0_nib) == sizeof(ggml_half) + QK8_0_NIB / 2, "block_q8_0_nib size mismatch");

// ── Device helpers ───────────────────────────────────────────────────────────

// Sign-extend a 4-bit nibble (unsigned, [0..15]) to int8 ([-8..7])
// Formula: (n ^ 8) - 8
//   n=0  → (0^8)-8=-8    n=8  → (8^8)-8=0
//   n=7  → (7^8)-8=7     n=15 → (15^8)-8=-1
static __device__ __forceinline__ int8_t nib_sign_ext(int n) {
    return (int8_t)((n ^ 8) - 8);
}

// Unpack 2 consecutive bytes of ra into one int32 of sign-extended int8×4
// for use with dp4a.  lo and hi each hold 2 nibbles (upper and lower halves).
//
// Elements extracted (for byte pair at position k within the ra buffer):
//   v[0] = sign_ext(upper nibble of lo) = element 4k
//   v[1] = sign_ext(lower nibble of lo) = element 4k+1
//   v[2] = sign_ext(upper nibble of hi) = element 4k+2
//   v[3] = sign_ext(lower nibble of hi) = element 4k+3
static __device__ __forceinline__ int nib_unpack4(uint8_t lo, uint8_t hi) {
    const int8_t n0 = nib_sign_ext(lo >> 4);
    const int8_t n1 = nib_sign_ext(lo & 0xF);
    const int8_t n2 = nib_sign_ext(hi >> 4);
    const int8_t n3 = nib_sign_ext(hi & 0xF);
    return (uint8_t)n0 | ((uint8_t)n1 << 8) | ((uint8_t)n2 << 16) | ((uint8_t)n3 << 24);
}

// ── vec_dot ──────────────────────────────────────────────────────────────────
//
// Compatible with the mul_mat_vec_q dispatch convention:
//   vbq   : base of block_q8_0_nib array for this row
//   bq8_1 : Q8_1 activation block at position kby (= kbx, since QK8_0_NIB=QK8_1)
//   kbx   : weight block index within the row
//   iqs   : int32 index within the ra buffer, ∈ {0,1,2,3}
//
// iqs=0 → ra[0..3]  → 8 nibbles covering elements  0..7 of this block
// iqs=1 → ra[4..7]  → elements  8..15
// iqs=2 → ra[8..11] → elements 16..23
// iqs=3 → ra[12..15]→ elements 24..31
//
// The matching Q8_1 activations are at int32 positions iqs*2 and iqs*2+1.
static __device__ __forceinline__ float vec_dot_q8_0_nib_q8_1(
    const void         * __restrict__ vbq,
    const block_q8_1   * __restrict__ bq8_1,
    const int          & kbx,
    const int          & iqs)
{
    const block_q8_0_nib * bq = (const block_q8_0_nib *) vbq + kbx;

    // Load 4 bytes of ra at int32 index iqs.
    // get_int_b2 (uint16_t reads) handles potential 2-byte-but-not-4-byte alignment
    // (sizeof(block_q8_0_nib)=18 → blocks not always 4B-aligned).
    const int ra_i32 = get_int_b2(bq->ra, iqs);

    // Unpack into two int8×4 packs for two dp4a calls:
    //   v0: elements {8*iqs+0, 1, 2, 3}  from bytes 0,1 of ra_i32
    //   v1: elements {8*iqs+4, 5, 6, 7}  from bytes 2,3 of ra_i32
    const int v0 = nib_unpack4((ra_i32 >>  0) & 0xFF, (ra_i32 >>  8) & 0xFF);
    const int v1 = nib_unpack4((ra_i32 >> 16) & 0xFF, (ra_i32 >> 24) & 0xFF);

    // Q8_1 activations at the matching element positions.
    // get_int_b4 is valid: bq8_1->qs is at offset +4 from block start,
    // and sizeof(block_q8_1)=36, so qs is always 4B-aligned.
    const int u0 = get_int_b4(bq8_1->qs, iqs * 2);      // elements 8*iqs+0..3
    const int u1 = get_int_b4(bq8_1->qs, iqs * 2 + 1);  // elements 8*iqs+4..7

    const int sumi = ggml_cuda_dp4a(v0, u0, ggml_cuda_dp4a(v1, u1, 0));

    // draft_scale × Q8_1 activation scale × integer dot product
    return __half2float(bq->d) * __low2float(bq8_1->ds) * (float)sumi;
}

// (No additional declarations needed; included directly by mmvq.cu for dispatch.)

// ── Host entry point ─────────────────────────────────────────────────────────

// Launch the nibble-draft GEMV.
//   weights     : device pointer to [n_rows × blocks_per_row] block_q8_0_nib
//   activations : device pointer to [blocks_per_row] block_q8_1 (the input vector)
//   dst         : device pointer to [n_rows] float output
//   ncols_x     : K (must be divisible by QK8_0_NIB = 32)
void ggml_cuda_mul_mat_vec_q8_0_nib(
    const block_q8_0_nib * weights,
    const block_q8_1     * activations,
    float                * dst,
    int n_rows, int ncols_x,
    cudaStream_t stream);

// ── CPU transform ─────────────────────────────────────────────────────────────

// Transform n_blocks Q8_0 blocks → block_q8_0_nib blocks (CPU, single-threaded).
// dst must have room for n_blocks × sizeof(block_q8_0_nib) bytes.
// Returns bytes written.
size_t transform_q8_0_to_nib(
    const block_q8_0    * src,
    block_q8_0_nib      * dst,
    int                   n_blocks);
