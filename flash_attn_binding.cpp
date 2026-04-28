/*
 * flash_attn_binding.cpp
 * ══════════════════════════════════════════════════════════════════════
 * PyTorch C++ extension binding.
 * Bridges Python → our CUDA kernels.
 *
 * After building with setup.py, use from Python as:
 *   import flash_attn_cuda
 *   O, L = flash_attn_cuda.forward(Q, K, V)
 * ══════════════════════════════════════════════════════════════════════
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

/* ── forward declarations (defined in flash_attn_kernels.cu) ─────── */
void launch_flash_attention(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    int B, int H, int N, int d,
    cudaStream_t stream);

void launch_naive_attention(
    const float* Q, const float* K, const float* V,
    float* O, float* S,
    int B, int H, int N, int d,
    cudaStream_t stream);


/* ══════════════════════════════════════════════════════════════════════
   flash_attention_forward
   Input tensors must be float32, CUDA, contiguous, shape [B, H, N, d].
   Returns: [O, L] where O is [B,H,N,d] and L is [B,H,N] (logsumexp).
   ══════════════════════════════════════════════════════════════════════ */
std::vector<torch::Tensor> flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V)
{
    /* ── input checks ─────────────────────────────────────────── */
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Q.dim() == 4, "Expected Q shape [B, H, N, d]");

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    /* ── allocate outputs ─────────────────────────────────────── */
    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({B, H, N}, Q.options());

    /* ── get current stream (respects PyTorch's stream context) ─ */
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_flash_attention(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), L.data_ptr<float>(),
        B, H, N, d, stream);

    return {O, L};
}


/* ══════════════════════════════════════════════════════════════════════
   naive_attention_forward (for benchmarking / correctness comparison)
   ══════════════════════════════════════════════════════════════════════ */
torch::Tensor naive_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V)
{
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(Q.dim() == 4, "Expected Q shape [B, H, N, d]");

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    auto O = torch::zeros_like(Q);
    /* scratch space for N×N score matrix */
    auto S = torch::zeros({B * H, N, N}, Q.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_naive_attention(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), S.data_ptr<float>(),
        B, H, N, d, stream);

    return O;
}


/* ── pybind11 module ─────────────────────────────────────────────── */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Simplified Flash Attention CUDA extension";

    m.def("forward",
          &flash_attention_forward,
          "Flash Attention forward pass (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"));

    m.def("naive_forward",
          &naive_attention_forward,
          "Naive attention forward pass for comparison (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"));
}
