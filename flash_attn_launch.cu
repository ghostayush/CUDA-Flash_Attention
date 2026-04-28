/*
 * flash_attn_launch.cu
 * ══════════════════════════════════════════════════════════════════════
 * Launcher wrapper functions called by flash_attn_binding.cpp.
 * Keeps kernel launch configuration in one place.
 * ══════════════════════════════════════════════════════════════════════
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define Br       16
#define Bc       16
#define HEAD_DIM 64

/* ── forward declarations of kernels (defined in flash_attn_kernels.cu) */
__global__ void flash_attention_fwd(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    int N, int d, int num_heads);

__global__ void naive_attention(
    const float* Q, const float* K, const float* V,
    float* O, float* S, int N, int d);

__global__ void naive_softmax_v(
    const float* S, const float* V, float* O, int N, int d);


void launch_flash_attention(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    int B, int H, int N, int d,
    cudaStream_t stream)
{
    int q_tiles  = (N + Br - 1) / Br;
    int total_bh = B * H;

    dim3 grid(q_tiles, total_bh);
    dim3 block(Br * Bc);

    size_t smem = (size_t)(Br*d + Bc*d + Bc*d + Br*Bc) * sizeof(float);

    flash_attention_fwd<<<grid, block, smem, stream>>>(
        Q, K, V, O, L, N, d, H);
}


void launch_naive_attention(
    const float* Q, const float* K, const float* V,
    float* O, float* S,
    int B, int H, int N, int d,
    cudaStream_t stream)
{
    int total_bh = B * H;
    dim3 block_qk(16, 16);
    dim3 grid_qk((N+15)/16, (N+15)/16);

    for (int bh = 0; bh < total_bh; ++bh) {
        const float* q = Q + bh * N * d;
        const float* k = K + bh * N * d;
        const float* v = V + bh * N * d;
        float* o = O + bh * N * d;
        float* s = S + bh * N * N;

        naive_attention<<<grid_qk, block_qk, 0, stream>>>(q, k, v, o, s, N, d);
        naive_softmax_v<<<N, 64, 0, stream>>>(s, v, o, N, d);
    }
}
