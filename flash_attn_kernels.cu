/*
 * flash_attn_kernels.cu
 * ══════════════════════════════════════════════════════════════════════
 * Project 2: Simplified Flash Attention
 *
 * Three kernels implemented:
 *   1. naive_attention        – computes full N×N attention matrix in HBM
 *   2. online_softmax         – one-pass softmax with warp-shuffle reduction
 *   3. flash_attention_fwd    – tiled attention: never materialises N×N matrix
 *
 * Key paper: "FlashAttention: Fast and Memory-Efficient Exact Attention
 *             with IO-Awareness" — Dao et al., NeurIPS 2022
 *
 * Compile (standalone benchmark):
 *   nvcc -O2 -arch=sm_75 --maxrregcount=64 flash_attn_kernels.cu \
 *        -lcublas -o flash_bench
 * ══════════════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* ── tuneable constants ──────────────────────────────────────────────── */
#define Br      16   /* tile rows for Q (output tile height)              */
#define Bc      16   /* tile cols for K/V                                  */
#define HEAD_DIM 64  /* attention head dimension (d_k = d_v)              */
#define WARP_SIZE 32

/* ── error macros ────────────────────────────────────────────────────── */
#define CUDA_CHECK(x) do { \
    cudaError_t e=(x); \
    if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
} while(0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t s=(x); \
    if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %s:%d code=%d\n",__FILE__,__LINE__,(int)s);exit(1);} \
} while(0)

/* ══════════════════════════════════════════════════════════════════════
   KERNEL 1 — Naive Attention
   Computes: O = softmax(Q K^T / sqrt(d)) V
   Each thread computes one element of the output O[i, j].
   Problem: materialises the full [seq_len × seq_len] attention matrix
            in global memory → O(N²) memory, becomes the bottleneck fast.
   ══════════════════════════════════════════════════════════════════════ */
__global__ void naive_attention(
    const float* __restrict__ Q,   /* [N, d] */
    const float* __restrict__ K,   /* [N, d] */
    const float* __restrict__ V,   /* [N, d] */
    float*       __restrict__ O,   /* [N, d] output */
    float*       __restrict__ S,   /* [N, N] scratch — the problem */
    int N, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  /* query row   */
    int j = blockIdx.y * blockDim.y + threadIdx.y;  /* key   col   */
    if (i >= N || j >= N) return;

    /* Step 1: compute S[i,j] = Q[i,:] · K[j,:] / sqrt(d) */
    float scale = rsqrtf((float)d);
    float s = 0.0f;
    for (int k = 0; k < d; ++k)
        s += Q[i * d + k] * K[j * d + k];
    S[i * N + j] = s * scale;
}

/* Separate pass: row-wise softmax on S, then multiply by V */
__global__ void naive_softmax_v(
    const float* __restrict__ S,   /* [N, N] */
    const float* __restrict__ V,   /* [N, d] */
    float*       __restrict__ O,   /* [N, d] */
    int N, int d)
{
    int i = blockIdx.x;  /* one block per query row */
    if (i >= N) return;

    /* find row max (for numerical stability) */
    float row_max = -FLT_MAX;
    for (int j = 0; j < N; ++j)
        row_max = fmaxf(row_max, S[i * N + j]);

    /* softmax denominator */
    float denom = 0.0f;
    for (int j = 0; j < N; ++j)
        denom += expf(S[i * N + j] - row_max);

    /* weighted sum over V */
    for (int k = threadIdx.x; k < d; k += blockDim.x) {
        float out = 0.0f;
        for (int j = 0; j < N; ++j) {
            float p = expf(S[i * N + j] - row_max) / denom;
            out += p * V[j * d + k];
        }
        O[i * d + k] = out;
    }
}


/* ══════════════════════════════════════════════════════════════════════
   KERNEL 2 — Online Softmax (standalone utility, not used in flash attn)
   One-pass numerically stable softmax using warp-shuffle reduction.
   Avoids the two-pass (find max, then compute) approach.

   Algorithm (Welford-style):
     Maintain running (m, l) = (max_so_far, normaliser_so_far).
     For each new value x:
       m_new = max(m, x)
       l_new = l * exp(m - m_new) + exp(x - m_new)
     Final: p_i = exp(x_i - m_final) / l_final
   ══════════════════════════════════════════════════════════════════════ */
__global__ void online_softmax(
    const float* __restrict__ in,   /* [B, N] */
    float*       __restrict__ out,  /* [B, N] */
    int B, int N)
{
    int batch = blockIdx.x;
    if (batch >= B) return;

    const float* row_in  = in  + batch * N;
    float*       row_out = out + batch * N;

    /* ── warp-parallel running (max, sum) over the row ─────────── */
    float m = -FLT_MAX;   /* running max   */
    float l = 0.0f;       /* running denom */

    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float x    = row_in[j];
        float m_new = fmaxf(m, x);
        l = l * expf(m - m_new) + expf(x - m_new);
        m = m_new;
    }

    /* ── warp-level reduction: get global max and sum ───────────── */
    /* reduce max across warp */
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xFFFFFFFF, m, offset);
        float l2 = __shfl_down_sync(0xFFFFFFFF, l, offset);
        float m_new = fmaxf(m, m2);
        l = l * expf(m - m_new) + l2 * expf(m2 - m_new);
        m = m_new;
    }

    /* thread 0 in warp holds global (m, l) — broadcast via shared mem */
    __shared__ float smem_m[1], smem_l[1];
    if (threadIdx.x == 0) { smem_m[0] = m; smem_l[0] = l; }
    __syncthreads();
    float global_m = smem_m[0];
    float global_l = smem_l[0];

    /* ── write normalised softmax output ────────────────────────── */
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        row_out[j] = expf(row_in[j] - global_m) / global_l;
}


/* ══════════════════════════════════════════════════════════════════════
   KERNEL 3 — Flash Attention Forward Pass
   ─────────────────────────────────────────────────────────────────────
   Core idea:
     Tile Q into blocks of Br rows.
     For each Q tile, loop over all K/V tiles (Bc columns).
     Maintain running (m_i, l_i, O_i) without ever writing the N×N
     attention matrix to HBM.

   SRAM usage per block:
     sQ:  Br × d   floats
     sK:  Bc × d   floats
     sV:  Bc × d   floats
     sS:  Br × Bc  floats  (one tile of scores, reused)
     Total: (2*Br*d + 2*Bc*d + Br*Bc) × 4 bytes
     With Br=Bc=16, d=64: (2*16*64 + 2*16*64 + 16*16) * 4 = 34 KB
     → fits in T4's 48 KB shared mem per SM.

   One block handles one (batch, head, Br-sized Q tile).
   gridDim = (N/Br, batch*heads).
   ══════════════════════════════════════════════════════════════════════ */
__global__ void flash_attention_fwd(
    const float* __restrict__ Q,    /* [B, H, N, d] — row major */
    const float* __restrict__ K,    /* [B, H, N, d] */
    const float* __restrict__ V,    /* [B, H, N, d] */
    float*       __restrict__ O,    /* [B, H, N, d] */
    float*       __restrict__ L,    /* [B, H, N]    logsumexp (for backward) */
    int N, int d, int num_heads)
{
    /* ── identify which (batch, head, Q-tile) this block handles ── */
    int bh   = blockIdx.y;                    /* combined batch*head index  */
    int q_tile = blockIdx.x;                  /* which Br-sized tile of Q   */
    int tx   = threadIdx.x;                   /* 0 … Br*Bc - 1              */

    int row_in_tile = tx / Bc;               /* 0 … Br-1 within tile        */
    int col_in_tile = tx % Bc;               /* 0 … Bc-1 within tile        */

    int q_row = q_tile * Br + row_in_tile;   /* absolute query row in [0,N) */

    /* offset to this (batch, head) in the 4-D tensor */
    int bh_offset = bh * N * d;

    /* ── shared memory layout ───────────────────────────────────── */
    extern __shared__ float smem[];
    float* sQ = smem;                          /* [Br, d]  */
    float* sK = sQ + Br * d;                   /* [Bc, d]  */
    float* sV = sK + Bc * d;                   /* [Bc, d]  */
    float* sS = sV + Bc * d;                   /* [Br, Bc] */

    /* ── load Q tile for this block (reused across all K/V tiles) ─ */
    /* Each thread loads one element of the Q tile */
    for (int elem = tx; elem < Br * d; elem += blockDim.x) {
        int r = elem / d, c = elem % d;
        int global_row = q_tile * Br + r;
        sQ[r * d + c] = (global_row < N) ? Q[bh_offset + global_row * d + c] : 0.0f;
    }
    __syncthreads();

    /* ── per-row running statistics (one per query row) ─────────── */
    /* Each thread in the row_in_tile dimension tracks its own row.  */
    float m_i = -FLT_MAX;   /* running max of scores for row q_row  */
    float l_i = 0.0f;        /* running softmax denominator           */
    float O_i[HEAD_DIM];     /* running weighted sum (output row)     */
    #pragma unroll
    for (int c = 0; c < HEAD_DIM; ++c) O_i[c] = 0.0f;

    int num_kv_tiles = (N + Bc - 1) / Bc;

    /* ── outer loop: iterate over K/V tiles ─────────────────────── */
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {

        /* load K tile */
        for (int elem = tx; elem < Bc * d; elem += blockDim.x) {
            int r = elem / d, c = elem % d;
            int global_row = kv_tile * Bc + r;
            sK[r * d + c] = (global_row < N) ? K[bh_offset + global_row * d + c] : 0.0f;
        }
        /* load V tile */
        for (int elem = tx; elem < Bc * d; elem += blockDim.x) {
            int r = elem / d, c = elem % d;
            int global_row = kv_tile * Bc + r;
            sV[r * d + c] = (global_row < N) ? V[bh_offset + global_row * d + c] : 0.0f;
        }
        __syncthreads();

        /* ── compute scores S[row_in_tile, col_in_tile] ─────────── */
        if (row_in_tile < Br && col_in_tile < Bc) {
            float scale = rsqrtf((float)d);
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < HEAD_DIM; ++k)
                s += sQ[row_in_tile * d + k] * sK[col_in_tile * d + k];
            sS[row_in_tile * Bc + col_in_tile] = s * scale;
        }
        __syncthreads();

        /* ── update running (m_i, l_i, O_i) for each query row ──── */
        /*    Only the "row master" threads (col_in_tile == 0) could
         *    do this serially, but we let each thread handle its own
         *    query row to stay simple. row_in_tile uniquely identifies
         *    the query row, so threads with same row_in_tile agree.   */
        if (col_in_tile == 0 && q_row < N) {
            /* find tile max */
            float m_tile = -FLT_MAX;
            for (int j = 0; j < Bc; ++j) {
                int kv_col = kv_tile * Bc + j;
                if (kv_col < N)
                    m_tile = fmaxf(m_tile, sS[row_in_tile * Bc + j]);
            }

            /* new global max */
            float m_new = fmaxf(m_i, m_tile);

            /* scale existing O and l by exp(m_i - m_new) */
            float scale_old = expf(m_i - m_new);
            l_i *= scale_old;
            #pragma unroll
            for (int c = 0; c < HEAD_DIM; ++c)
                O_i[c] *= scale_old;

            /* accumulate new P*V contribution */
            float l_tile = 0.0f;
            for (int j = 0; j < Bc; ++j) {
                int kv_col = kv_tile * Bc + j;
                if (kv_col < N) {
                    float p = expf(sS[row_in_tile * Bc + j] - m_new);
                    l_tile += p;
                    #pragma unroll
                    for (int c = 0; c < HEAD_DIM; ++c)
                        O_i[c] += p * sV[j * d + c];
                }
            }

            l_i += l_tile;
            m_i  = m_new;
        }
        __syncthreads();
    }

    /* ── write normalised output ─────────────────────────────────── */
    if (col_in_tile == 0 && q_row < N) {
        float inv_l = 1.0f / l_i;
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; ++c)
            O[bh_offset + q_row * d + c] = O_i[c] * inv_l;

        /* save logsumexp for backward pass: log(l_i) + m_i */
        if (L != NULL)
            L[bh * N + q_row] = logf(l_i) + m_i;
    }
}


/* ══════════════════════════════════════════════════════════════════════
   TIMING HELPERS
   ══════════════════════════════════════════════════════════════════════ */
static float gpu_ms(cudaEvent_t s, cudaEvent_t e) {
    float ms; CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); return ms;
}

static float max_err(const float* a, const float* b, int n) {
    float me = 0.f;
    for (int i = 0; i < n; ++i) {
        float e = fabsf(a[i] - b[i]);
        if (e > me) me = e;
    }
    return me;
}

static double compute_bandwidth_GB(int N, int d, float ms) {
    /* HBM bytes read by naive: Q(Nd) + K(Nd) + V(Nd) + S(N²) read+write */
    double naive_bytes = 4.0 * (2.0 * N * d + N * N) * sizeof(float);
    /* HBM bytes read by flash: Q(Nd) + K_tiles(Nd) + V_tiles(Nd) + O(Nd) */
    double flash_bytes = 4.0 * 4.0 * N * d * sizeof(float);
    (void)naive_bytes;
    return flash_bytes / (ms * 1e6);  /* GB/s effective bandwidth */
}


/* ══════════════════════════════════════════════════════════════════════
   BENCHMARK — one (N, d) configuration
   ══════════════════════════════════════════════════════════════════════ */
static void benchmark(int N, int d, int B, int H,
                      cudaEvent_t ev_s, cudaEvent_t ev_e,
                      int warmup, int reps,
                      float* naive_ms_out, float* flash_ms_out,
                      FILE* csv)
{
    size_t qkv_bytes = (size_t)B * H * N * d * sizeof(float);
    size_t s_bytes   = (size_t)B * H * N * N * sizeof(float);
    size_t l_bytes   = (size_t)B * H * N     * sizeof(float);

    /* ── host alloc + random init ──────────────────────────────── */
    float *hQ=(float*)malloc(qkv_bytes), *hK=(float*)malloc(qkv_bytes);
    float *hV=(float*)malloc(qkv_bytes);
    float *hO_naive=(float*)malloc(qkv_bytes);
    float *hO_flash=(float*)malloc(qkv_bytes);

    for (int i = 0; i < (int)(B*H*N*d); ++i) {
        hQ[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
        hK[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
        hV[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    }

    /* ── device alloc ──────────────────────────────────────────── */
    float *dQ,*dK,*dV,*dO,*dS,*dL;
    CUDA_CHECK(cudaMalloc(&dQ, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&dK, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&dV, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&dO, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&dS, s_bytes));
    CUDA_CHECK(cudaMalloc(&dL, l_bytes));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, qkv_bytes, cudaMemcpyHostToDevice));

    float ms;

    /* ── NAIVE attention ───────────────────────────────────────── */
    {
        int total_seq = B * H;
        dim3 block_qk(16, 16);
        dim3 grid_qk((N+15)/16, (N+15)/16);

        for (int i = 0; i < warmup; ++i) {
            for (int bh = 0; bh < total_seq; ++bh) {
                naive_attention<<<grid_qk, block_qk>>>(
                    dQ + bh*N*d, dK + bh*N*d, dV + bh*N*d,
                    dO + bh*N*d, dS + bh*N*N, N, d);
                naive_softmax_v<<<N, 64>>>(
                    dS + bh*N*N, dV + bh*N*d, dO + bh*N*d, N, d);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(ev_s));
        for (int r = 0; r < reps; ++r) {
            for (int bh = 0; bh < total_seq; ++bh) {
                naive_attention<<<grid_qk, block_qk>>>(
                    dQ + bh*N*d, dK + bh*N*d, dV + bh*N*d,
                    dO + bh*N*d, dS + bh*N*N, N, d);
                naive_softmax_v<<<N, 64>>>(
                    dS + bh*N*N, dV + bh*N*d, dO + bh*N*d, N, d);
            }
        }
        CUDA_CHECK(cudaEventRecord(ev_e));
        ms = gpu_ms(ev_s, ev_e) / reps;
        *naive_ms_out = ms;
        CUDA_CHECK(cudaMemcpy(hO_naive, dO, qkv_bytes, cudaMemcpyDeviceToHost));

        double gflops = (2.0*B*H*(N*N*d + N*N*d)) / (ms * 1e6);
        printf("  Naive attention  : %8.3f ms   %.1f GFLOP/s   mem≈%.0f MB\n",
               ms, gflops, (double)s_bytes / 1e6);
    }

    /* ── FLASH attention ───────────────────────────────────────── */
    {
        int q_tiles = (N + Br - 1) / Br;
        int total_bh = B * H;
        dim3 grid(q_tiles, total_bh);
        dim3 block(Br * Bc);   /* Br*Bc threads per block */

        /* shared mem: sQ[Br*d] + sK[Bc*d] + sV[Bc*d] + sS[Br*Bc] */
        size_t smem = (size_t)(Br*d + Bc*d + Bc*d + Br*Bc) * sizeof(float);

        for (int i = 0; i < warmup; ++i)
            flash_attention_fwd<<<grid, block, smem>>>(
                dQ, dK, dV, dO, dL, N, d, H);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(ev_s));
        for (int r = 0; r < reps; ++r)
            flash_attention_fwd<<<grid, block, smem>>>(
                dQ, dK, dV, dO, dL, N, d, H);
        CUDA_CHECK(cudaEventRecord(ev_e));
        ms = gpu_ms(ev_s, ev_e) / reps;
        *flash_ms_out = ms;
        CUDA_CHECK(cudaMemcpy(hO_flash, dO, qkv_bytes, cudaMemcpyDeviceToHost));

        double gflops = (2.0*B*H*(N*N*d + N*N*d)) / (ms * 1e6);
        float err = max_err(hO_naive, hO_flash, B*H*N*d);
        printf("  Flash attention  : %8.3f ms   %.1f GFLOP/s   mem=O(Nd)   err=%.2e\n",
               ms, gflops, err);

        if (err > 1e-2f)
            printf("  WARNING: max error %.2e exceeds threshold — check kernel\n", err);
    }

    float speedup = *naive_ms_out / *flash_ms_out;
    float mem_naive_MB = (float)s_bytes / 1e6f;
    printf("  Speedup          : %.2fx   Memory saved: %.0f MB → ~%d KB (SRAM)\n\n",
           speedup, mem_naive_MB,
           (int)((Br*d + Bc*d + Bc*d + Br*Bc) * sizeof(float) / 1024));

    if (csv)
        fprintf(csv, "%d,%d,%d,%.4f,%.4f,%.3f\n",
                N, d, B*H, *naive_ms_out, *flash_ms_out, speedup);

    /* cleanup */
    CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dK));
    CUDA_CHECK(cudaFree(dV)); CUDA_CHECK(cudaFree(dO));
    CUDA_CHECK(cudaFree(dS)); CUDA_CHECK(cudaFree(dL));
    free(hQ); free(hK); free(hV); free(hO_naive); free(hO_flash);
}


/* ══════════════════════════════════════════════════════════════════════
   MAIN
   ══════════════════════════════════════════════════════════════════════ */
int main(void) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("══════════════════════════════════════════════════════\n");
    printf("GPU : %s\n", prop.name);
    printf("SMs : %d   Shared mem/block: %.0f KB\n",
           prop.multiProcessorCount, prop.sharedMemPerBlock/1024.0f);
    printf("Tile: Br=%d  Bc=%d  d=%d\n", Br, Bc, HEAD_DIM);
    printf("SRAM/block: %.1f KB\n",
           (Br*HEAD_DIM + Bc*HEAD_DIM + Bc*HEAD_DIM + Br*Bc)*4.0f/1024.0f);
    printf("══════════════════════════════════════════════════════\n\n");

    FILE* csv = fopen("flash_results.csv", "w");
    fprintf(csv, "N,d,BH,naive_ms,flash_ms,speedup\n");

    cudaEvent_t ev_s, ev_e;
    CUDA_CHECK(cudaEventCreate(&ev_s));
    CUDA_CHECK(cudaEventCreate(&ev_e));
    srand(42);

    int B = 1, H = 8;  /* batch=1, 8 heads — vary as needed */
    int warmup = 3, reps = 20;

    /* sweep sequence lengths */
    int seq_lens[] = {128, 256, 512, 1024};
    int d = HEAD_DIM;

    for (int i = 0; i < 4; ++i) {
        int N = seq_lens[i];
        printf("── N=%d, d=%d, B=%d, H=%d ──────────────────────────\n",
               N, d, B, H);
        float naive_ms, flash_ms;
        benchmark(N, d, B, H, ev_s, ev_e, warmup, reps,
                  &naive_ms, &flash_ms, csv);
    }

    fclose(csv);
    CUDA_CHECK(cudaEventDestroy(ev_s));
    CUDA_CHECK(cudaEventDestroy(ev_e));
    printf("Results saved to flash_results.csv\n");
    return 0;
}
