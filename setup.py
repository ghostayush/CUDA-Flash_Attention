"""
setup.py
Build the flash attention PyTorch CUDA extension.

Usage:
    pip install -e .          # install in editable mode (dev)
    python setup.py build_ext --inplace   # build .so in current dir

After installation:
    import flash_attn_cuda
    O, L = flash_attn_cuda.forward(Q, K, V)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Detect GPU architecture from installed torch
def get_cuda_arch():
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        major, minor = capability
        return [f"sm_{major}{minor}"]
    # Default: T4 (sm_75)
    return ["sm_75"]

arch_flags = get_cuda_arch()
print(f"Building for GPU architecture: {arch_flags}")

nvcc_flags = [
    "-O2",
    "--maxrregcount=64",
    "-std=c++17",
    "--use_fast_math",          # use __expf, __rsqrtf etc — faster, slight precision loss
] + [f"-arch={a}" for a in arch_flags]

setup(
    name="flash_attn_cuda",
    version="0.1.0",
    description="Simplified Flash Attention CUDA extension (educational)",
    ext_modules=[
        CUDAExtension(
            name="flash_attn_cuda",
            sources=[
                "flash_attn_binding.cpp",
                "flash_attn_kernels.cu",
                "flash_attn_launch.cu",
            ],
            extra_compile_args={
                "cxx":  ["-O2", "-std=c++17"],
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
