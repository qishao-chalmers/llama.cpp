/*
 * bench_nibble_gemv.cu  —  Microbenchmark: Q8_0 GEMV vs nibble-split draft GEMV
 *
 * Measures achieved HBM bandwidth in TWO regimes:
 *
 *   [L2 regime]  Single layer  (25 MB nibble / 50 MB Q8_0): data fits in L2 cache.
 *                Results reflect L2-served throughput — NOT representative of decode.
 *
 *   [HBM regime] N_LAYERS=36 layers streamed in sequence (900 MB nibble / 1.8 GB Q8_0):
 *                exceeds 50 MB L2 → data streams from HBM each pass.
 *                Matches real decode: 7.4 GB weights >> L2, always HBM-bound.
 *
 * Matrix per layer: M=12288 (ffn_dim), K=4096 (d_model) — Qwen3-8B FFN gate/up.
 * Lesson from v1: interleaved block layout (ILV) was WORSE because its 56 MB
 * exceeded L2, causing more misses than the flat 25 MB nibble array.
 * Flat layout wins: smaller footprint → better L2 fit → less total HBM traffic.
 *
 * Build:
 *   nvcc -O3 -arch=sm_90 --compiler-options -Wall \
 *        -o bench_nibble_gemv bench_nibble_gemv.cu -lm
 *
 * Pass criteria (research/context/split_weight_precision.md §11):
 *   [HBM regime] nibble_draft η >= 0.80 × q8_baseline η → wall-clock speedup >= 1.6×
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// ── Configuration ─────────────────────────────────────────────────────────────
static const int M        = 12288;   // output features (ffn_dim)
static const int K        = 4096;    // input  features (d_model)
static const int QK       = 32;      // Q8_0 block size
static const int NB       = K / QK;  // 128 blocks per row
static const int WARPS    = 8;       // warps per thread block (= rows per block)
static const int N_LAYERS = 36;      // layers to stream per iteration (HBM regime)
static const int WARMUP   = 20;
static const int ITERS    = 100;     // fewer iters: each iter streams N_LAYERS layers

// ── Error check ───────────────────────────────────────────────────────────────
#define CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_e)); \
            exit(1); \
        } \
    } while (0)

// ── Memory layout (flat, separate buffers) ────────────────────────────────────
//
//  w  [M, K]      int8   Q8_0 weights, row-major
//  s  [M, NB]     fp16   block scales
//  ra [M, K/2]    uint8  upper nibbles of each element, packed 2-per-byte, row-major
//  rb [M, K/2]    uint8  lower nibbles
//  ds [M, NB]     fp16   draft_scale = full_scale × 16
//
//  Nibble packing for element i in row r:
//    byte index  = i / 2         within ra/rb for that row
//    nibble side = upper (bits [7:4]) if i%2==0, lower (bits [3:0]) if i%2==1
//
//  Access per warp per block (32 lanes):
//    Q8_0:  lane l reads int8 at offset b*32+l  → 32 consecutive bytes
//    nibble: lane l reads uint8 at offset b*16+l/2 → 16 consecutive bytes
//             (lanes 2l and 2l+1 share byte l — warp broadcast, no extra transactions)
//    Both issue one 128-byte L2/HBM cache line per block per warp.
//    With data >> L2: fewer bytes → fewer cache misses → meaningful speedup.

// ── Kernel 1: Q8_0 baseline ───────────────────────────────────────────────────
__global__ void k_q8_baseline(
    const int8_t* __restrict__ w,
    const half*   __restrict__ s,
    const half*   __restrict__ x,
    float*        __restrict__ y,
    int n_rows
) {
    const int row  = blockIdx.x * WARPS + threadIdx.y;
    if (row >= n_rows) return;
    const int lane = threadIdx.x;

    const int8_t* rw = w + (size_t)row * K;
    const half*   rs = s + (size_t)row * NB;

    float acc = 0.f;
#pragma unroll 4
    for (int b = 0; b < NB; b++) {
        const float sc = __half2float(__ldg(&rs[b]));
        const float wi = (float)__ldg(&rw[b * QK + lane]);
        const float xi = __half2float(__ldg(&x[b * QK + lane]));
        acc += sc * wi * xi;
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) y[row] = acc;
}

// ── Kernel 2: Nibble draft (Region A only) ────────────────────────────────────
__global__ void k_nibble_draft(
    const uint8_t* __restrict__ ra,
    const half*    __restrict__ ds,
    const half*    __restrict__ x,
    float*         __restrict__ y,
    int n_rows
) {
    const int row  = blockIdx.x * WARPS + threadIdx.y;
    if (row >= n_rows) return;
    const int lane = threadIdx.x;

    const uint8_t* rra = ra + (size_t)row * (K / 2);
    const half*    rds = ds + (size_t)row * NB;

    float acc = 0.f;
#pragma unroll 4
    for (int b = 0; b < NB; b++) {
        const float dsc    = __half2float(__ldg(&rds[b]));
        const uint8_t pack = __ldg(&rra[b * (QK / 2) + lane / 2]);
        const int nibble   = (lane & 1) ? (pack & 0xF) : (pack >> 4);
        const int snib     = (nibble ^ 8) - 8;   // sign-extend 4-bit → signed
        const float xi     = __half2float(__ldg(&x[b * QK + lane]));
        acc += dsc * (float)snib * xi;
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) y[row] = acc;
}

// ── Kernel 3: Nibble verify (Region A+B → reconstruct int8) ──────────────────
__global__ void k_nibble_verify(
    const uint8_t* __restrict__ ra,
    const uint8_t* __restrict__ rb,
    const half*    __restrict__ s,
    const half*    __restrict__ x,
    float*         __restrict__ y,
    int n_rows
) {
    const int row  = blockIdx.x * WARPS + threadIdx.y;
    if (row >= n_rows) return;
    const int lane = threadIdx.x;

    const uint8_t* rra = ra + (size_t)row * (K / 2);
    const uint8_t* rrb = rb + (size_t)row * (K / 2);
    const half*    rs  = s  + (size_t)row * NB;

    float acc = 0.f;
#pragma unroll 4
    for (int b = 0; b < NB; b++) {
        const float    sc  = __half2float(__ldg(&rs[b]));
        const int bi       = b * (QK / 2) + lane / 2;
        const uint8_t  pa  = __ldg(&rra[bi]);
        const uint8_t  pb  = __ldg(&rrb[bi]);
        const int na = (lane & 1) ? (pa & 0xF) : (pa >> 4);
        const int nb = (lane & 1) ? (pb & 0xF) : (pb >> 4);
        const int8_t  qi   = (int8_t)((na << 4) | nb);
        const float   xi   = __half2float(__ldg(&x[b * QK + lane]));
        acc += sc * (float)qi * xi;
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) y[row] = acc;
}

// ── CPU helpers ───────────────────────────────────────────────────────────────
static void build_nibble(
    const int8_t* w, const half* s,
    uint8_t* ra, uint8_t* rb, half* ds
) {
    for (int row = 0; row < M; row++) {
        for (int b = 0; b < NB; b++)
            ds[row * NB + b] = __float2half(__half2float(s[row * NB + b]) * 16.f);
        for (int i = 0; i < K; i += 2) {
            uint8_t q0 = (uint8_t)w[row * K + i];
            uint8_t q1 = (uint8_t)w[row * K + i + 1];
            int j = row * (K / 2) + i / 2;
            ra[j] = (q0 & 0xF0u) | (q1 >> 4);
            rb[j] = ((q0 & 0x0Fu) << 4) | (q1 & 0x0Fu);
        }
    }
}

static void cpu_gemv(
    const int8_t* w, const half* s, const half* x,
    double* y, int n_rows
) {
    for (int r = 0; r < n_rows; r++) {
        double acc = 0.0;
        for (int b = 0; b < NB; b++) {
            double sc = (double)__half2float(s[r * NB + b]);
            for (int i = 0; i < QK; i++)
                acc += sc * (double)w[r * K + b * QK + i]
                          * (double)__half2float(x[b * QK + i]);
        }
        y[r] = acc;
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(void) {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    // Correct BW formula for HBM (DDR): BW = 2 × SDR_clock × bus_bytes
    double peak_bw = 2.0 * prop.memoryClockRate * 1e3   // Hz
                   * (prop.memoryBusWidth / 8.0)         // bytes/transfer
                   / 1e9;                                 // → GB/s
    // Note: CUDA reports SDR clock in kHz; factor-of-2 gap vs vendor spec means
    // some H100 variants report half-rate. Use nvidia-smi for authoritative BW.

    printf("Device: %s\n", prop.name);
    printf("Peak HBM BW (CUDA): %.0f GB/s  "
           "(H100 SXM5 spec = 3350 GB/s — may differ due to clock reporting)\n",
           peak_bw);
    printf("L2 cache: %.0f MB\n", prop.l2CacheSize / 1e6);
    printf("Matrix per layer: M=%d  K=%d  QK=%d  NB=%d\n", M, K, QK, NB);

    // Data sizes per layer
    size_t sz_w   = (size_t)M * K;
    size_t sz_s   = (size_t)M * NB * sizeof(half);
    size_t sz_nib = (size_t)M * (K / 2);
    size_t sz_x   = (size_t)K * sizeof(half);
    size_t sz_y   = (size_t)M * sizeof(float);

    double mb_q8   = (double)(sz_w + sz_s) / 1e6;
    double mb_draft= (double)(sz_nib + sz_s) / 1e6;
    double mb_vfy  = (double)(sz_nib * 2 + sz_s) / 1e6;
    printf("\nData per layer:\n");
    printf("  Q8_0  (int8 weights + scales): %.1f MB\n", mb_q8);
    printf("  nibble draft (ra + ds):        %.1f MB  (%.0f%% of Q8_0)\n",
           mb_draft, mb_draft / mb_q8 * 100);
    printf("  nibble verify (ra+rb+s):       %.1f MB  (same as Q8_0)\n", mb_vfy);
    printf("  L2 cache: %.0f MB  →  ", prop.l2CacheSize / 1e6);
    printf("Q8 %s L2, nibble %s L2\n",
           mb_q8   < prop.l2CacheSize / 1e6 ? "fits in" : "exceeds",
           mb_draft< prop.l2CacheSize / 1e6 ? "fits in" : "exceeds");
    printf("  %d layers Q8_0:  %.0f MB  →  %s L2\n", N_LAYERS,
           N_LAYERS * mb_q8,
           N_LAYERS * mb_q8 < prop.l2CacheSize / 1e6 ? "fits in" : "exceeds (HBM-bound)");
    printf("  %d layers nibble: %.0f MB  →  %s L2\n\n", N_LAYERS,
           N_LAYERS * mb_draft,
           N_LAYERS * mb_draft < prop.l2CacheSize / 1e6 ? "fits in" : "exceeds (HBM-bound)");

    // ── Host alloc & init ONE layer (reuse same weights across N_LAYERS) ──────
    // For the HBM regime we allocate N_LAYERS separate buffers so the GPU can't
    // cache reuse them across layers (simulates real decode weight streaming).
    int8_t*  h_w  = (int8_t*) malloc(sz_w);
    half*    h_s  = (half*)   malloc(sz_s);
    uint8_t* h_ra = (uint8_t*)malloc(sz_nib);
    uint8_t* h_rb = (uint8_t*)malloc(sz_nib);
    half*    h_ds = (half*)   malloc(sz_s);
    half*    h_x  = (half*)   malloc(sz_x);
    double*  h_yref=(double*) malloc(64 * sizeof(double));
    float*   h_y   = (float*) malloc(sz_y);

    srand(42);
    for (size_t i = 0; i < sz_w;           i++) h_w[i] = (int8_t)(rand() % 256 - 128);
    for (size_t i = 0; i < sz_s/sizeof(half); i++)
        h_s[i] = __float2half((rand() % 100) / 1000.f + 0.001f);
    for (size_t i = 0; i < sz_x/sizeof(half); i++)
        h_x[i] = __float2half((rand() % 200 - 100) / 100.f);

    build_nibble(h_w, h_s, h_ra, h_rb, h_ds);
    cpu_gemv(h_w, h_s, h_x, h_yref, 64);

    // ── Device: single-layer buffers (L2 regime) ──────────────────────────────
    int8_t*  d_w;   CHECK(cudaMalloc(&d_w,  sz_w));
    half*    d_s;   CHECK(cudaMalloc(&d_s,  sz_s));
    uint8_t* d_ra;  CHECK(cudaMalloc(&d_ra, sz_nib));
    uint8_t* d_rb;  CHECK(cudaMalloc(&d_rb, sz_nib));
    half*    d_ds;  CHECK(cudaMalloc(&d_ds, sz_s));
    half*    d_x;   CHECK(cudaMalloc(&d_x,  sz_x));
    float*   d_y;   CHECK(cudaMalloc(&d_y,  sz_y));

    CHECK(cudaMemcpy(d_w,  h_w,  sz_w,  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_s,  h_s,  sz_s,  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ra, h_ra, sz_nib,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_rb, h_rb, sz_nib,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ds, h_ds, sz_s,  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x,  h_x,  sz_x,  cudaMemcpyHostToDevice));

    // ── Device: multi-layer buffers (HBM regime) ──────────────────────────────
    // Allocate N_LAYERS separate weight arrays so data >> L2
    int8_t**  dl_w  = (int8_t**) malloc(N_LAYERS * sizeof(void*));
    half**    dl_s  = (half**)   malloc(N_LAYERS * sizeof(void*));
    uint8_t** dl_ra = (uint8_t**)malloc(N_LAYERS * sizeof(void*));
    uint8_t** dl_rb = (uint8_t**)malloc(N_LAYERS * sizeof(void*));
    half**    dl_ds = (half**)   malloc(N_LAYERS * sizeof(void*));

    printf("Allocating %d-layer HBM buffers (%.0f MB Q8_0, %.0f MB nibble)...\n",
           N_LAYERS, N_LAYERS * mb_q8, N_LAYERS * mb_draft);
    for (int l = 0; l < N_LAYERS; l++) {
        CHECK(cudaMalloc(&dl_w[l],  sz_w));
        CHECK(cudaMalloc(&dl_s[l],  sz_s));
        CHECK(cudaMalloc(&dl_ra[l], sz_nib));
        CHECK(cudaMalloc(&dl_rb[l], sz_nib));
        CHECK(cudaMalloc(&dl_ds[l], sz_s));
        // Use same weights for all layers (correctness doesn't change)
        CHECK(cudaMemcpy(dl_w[l],  h_w,  sz_w,  cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dl_s[l],  h_s,  sz_s,  cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dl_ra[l], h_ra, sz_nib,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dl_rb[l], h_rb, sz_nib,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dl_ds[l], h_ds, sz_s,  cudaMemcpyHostToDevice));
    }

    // ── Launch config ─────────────────────────────────────────────────────────
    dim3 block(32, WARPS);
    dim3 grid((M + WARPS - 1) / WARPS);
    dim3 grid_check((64 + WARPS - 1) / WARPS);

    // ── Correctness (single layer) ────────────────────────────────────────────
    printf("\nCorrectness vs CPU fp64 (64 rows):\n");
    auto check = [&](const char* name, auto fn) {
        CHECK(cudaMemset(d_y, 0, sz_y));
        fn(grid_check);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_y, d_y, 64 * sizeof(float), cudaMemcpyDeviceToHost));
        double err = 0.0;
        for (int i = 0; i < 64; i++)
            err = fmax(err, fabs((double)h_y[i] - h_yref[i]));
        printf("  %-20s max_err = %.5f  %s\n", name, err,
               err < 0.1 ? "PASS" : "FAIL");
    };
    check("q8_baseline",   [&](dim3 g){ k_q8_baseline  <<<g,block>>>(d_w,d_s,d_x,d_y,64); });
    check("nibble_verify", [&](dim3 g){ k_nibble_verify <<<g,block>>>(d_ra,d_rb,d_s,d_x,d_y,64); });
    // Draft: approximate, just print
    CHECK(cudaMemset(d_y, 0, sz_y));
    k_nibble_draft<<<grid_check, block>>>(d_ra, d_ds, d_x, d_y, 64);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_y, d_y, 64 * sizeof(float), cudaMemcpyDeviceToHost));
    double derr = 0.0;
    for (int i = 0; i < 64; i++) derr = fmax(derr, fabs((double)h_y[i] - h_yref[i]));
    printf("  %-20s max_err = %.5f  (approx — lower precision expected)\n\n",
           "nibble_draft", derr);

    // ── Timing helpers ────────────────────────────────────────────────────────
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    struct Res { double gbs; double ms; };

    // event_ms must be defined before the lambdas that call it
    auto event_ms = [&](cudaEvent_t a, cudaEvent_t b) -> float {
        float ms; CHECK(cudaEventElapsedTime(&ms, a, b)); return ms;
    };

    auto bench_single = [&](const char* label, double nbytes, auto fn) -> Res {
        for (int i = 0; i < WARMUP * 5; i++) fn(d_w, d_s, d_ra, d_rb, d_ds);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < ITERS * 5; i++) fn(d_w, d_s, d_ra, d_rb, d_ds);
        CHECK(cudaEventRecord(ev1)); CHECK(cudaDeviceSynchronize());
        double ms  = (double)event_ms(ev0, ev1) / (ITERS * 5);
        double gbs = nbytes / ms / 1e6;
        printf("  %-20s %6.3f ms   %7.1f GB/s   (%.1f MB)\n", label, ms, gbs, nbytes/1e6);
        return {gbs, ms};
    };
    auto bench_multilayer = [&](const char* label, double nbytes_per_layer, auto fn) -> Res {
        // Each iter: stream through all N_LAYERS
        auto run_all = [&]{
            for (int l = 0; l < N_LAYERS; l++)
                fn(dl_w[l], dl_s[l], dl_ra[l], dl_rb[l], dl_ds[l]);
        };
        for (int i = 0; i < WARMUP; i++) run_all();
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < ITERS; i++) run_all();
        CHECK(cudaEventRecord(ev1)); CHECK(cudaDeviceSynchronize());
        double total_ms    = (double)event_ms(ev0, ev1);
        double ms_per_step = total_ms / ITERS;       // ms per full N_LAYERS pass
        double ms_per_layer= ms_per_step / N_LAYERS; // ms per single layer
        double gbs = nbytes_per_layer / ms_per_layer / 1e6;
        printf("  %-20s %6.3f ms/layer  %7.1f GB/s   (%.0f MB/layer, %d layers/step)\n",
               label, ms_per_layer, gbs, nbytes_per_layer/1e6, N_LAYERS);
        return {gbs, ms_per_layer};
    };

    // fn signature: (w, s, ra, rb, ds) → launches kernel ignoring unused args
    auto fn_q8  = [&](int8_t* w, half* s, uint8_t* ra, uint8_t* rb, half* ds){
        (void)ra; (void)rb; (void)ds;
        k_q8_baseline <<<grid,block>>>(w,  s,  d_x, d_y, M); };
    auto fn_draft= [&](int8_t* w, half* s, uint8_t* ra, uint8_t* rb, half* ds){
        (void)w; (void)s;
        k_nibble_draft<<<grid,block>>>(ra, ds, d_x, d_y, M); };
    auto fn_vfy  = [&](int8_t* w, half* s, uint8_t* ra, uint8_t* rb, half* ds){
        (void)w; (void)ds;
        k_nibble_verify<<<grid,block>>>(ra, rb, s, d_x, d_y, M); };

    printf("── L2 regime (single layer — data may fit in L2 cache) ────────────────\n");
    Res sl_q8    = bench_single("q8_baseline",   mb_q8   * 1e6, fn_q8);
    Res sl_draft = bench_single("nibble_draft",  mb_draft* 1e6, fn_draft);
    Res sl_vfy   = bench_single("nibble_verify", mb_vfy  * 1e6, fn_vfy);

    printf("\n── HBM regime (%d layers — %.0f MB Q8_0, %.0f MB nibble >> L2) ───────\n",
           N_LAYERS, N_LAYERS * mb_q8, N_LAYERS * mb_draft);
    Res ml_q8    = bench_multilayer("q8_baseline",   mb_q8   * 1e6, fn_q8);
    Res ml_draft = bench_multilayer("nibble_draft",  mb_draft* 1e6, fn_draft);
    Res ml_vfy   = bench_multilayer("nibble_verify", mb_vfy  * 1e6, fn_vfy);

    auto print_summary = [&](const char* regime, Res r_q8, Res r_draft, Res r_vfy) {
        double eta_d = r_draft.gbs / r_q8.gbs;
        double eta_v = r_vfy.gbs   / r_q8.gbs;
        double ws_d  = r_q8.ms / r_draft.ms;   // wall-clock speedup
        printf("\n  %s:\n", regime);
        printf("    q8_baseline:   %.1f GB/s  (100%% baseline)\n", r_q8.gbs);
        printf("    nibble_draft:  %.1f GB/s  η ratio=%.3f×  wall_speedup=%.2f×  %s\n",
               r_draft.gbs, eta_d, ws_d,
               eta_d >= 0.80 ? "PASS ✓" : "FAIL ✗ (< 0.80 threshold)");
        printf("    nibble_verify: %.1f GB/s  η ratio=%.3f×  (same bytes as q8, ~1.0 expected)\n",
               r_vfy.gbs, eta_v);
    };

    printf("\n════ Summary ════════════════════════════════════════════════════════════\n");
    print_summary("L2 regime (cached, not representative of decode)", sl_q8, sl_draft, sl_vfy);
    print_summary("HBM regime (streaming, matches real decode)",      ml_q8, ml_draft, ml_vfy);

    printf("\n  Pass criteria: [HBM regime] nibble_draft η >= 0.80 × q8_baseline η\n");
    double eta_hbm = ml_draft.gbs / ml_q8.gbs;
    double ws_hbm  = ml_q8.ms / ml_draft.ms;
    if (eta_hbm >= 0.80) {
        printf("  → PASS (%.3f)  Effective wall-clock speedup: %.2f×\n", eta_hbm, ws_hbm);
        printf("  → Proceed with full split-weight integration.\n");
    } else {
        printf("  → FAIL (%.3f < 0.80)  Wall-clock speedup: %.2f×\n", eta_hbm, ws_hbm);
        printf("  → Consider: shared-mem scale prefetch, warp-level nibble sharing,\n");
        printf("              or revisit go/no-go threshold given real decode context.\n");
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_w); cudaFree(d_s); cudaFree(d_ra); cudaFree(d_rb);
    cudaFree(d_ds); cudaFree(d_x); cudaFree(d_y);
    for (int l = 0; l < N_LAYERS; l++) {
        cudaFree(dl_w[l]); cudaFree(dl_s[l]); cudaFree(dl_ra[l]);
        cudaFree(dl_rb[l]); cudaFree(dl_ds[l]);
    }
    free(dl_w); free(dl_s); free(dl_ra); free(dl_rb); free(dl_ds);
    free(h_w); free(h_s); free(h_ra); free(h_rb);
    free(h_ds); free(h_x); free(h_yref); free(h_y);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return 0;
}
