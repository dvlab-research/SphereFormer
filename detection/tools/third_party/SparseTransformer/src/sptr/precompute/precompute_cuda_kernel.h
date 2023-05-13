#ifndef PRECOMPUTE_CUDA_KERNEL
#define PRECOMPUTE_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void precompute_all_cuda(int N, int n, const unsigned int n_max, at::Tensor counts_tensor, at::Tensor offsets_tensor, at::Tensor sq_offsets_tensor, at::Tensor index_0_offsets_tensor, at::Tensor index_1_offsets_tensor, at::Tensor index_0, at::Tensor index_1);

#ifdef __cplusplus
extern "C" {
#endif

void precompute_all_cuda_launcher(int N, int n, const unsigned int n_max, const int *counts, const int *offsets, const int *sq_offsets, int *index_0_offsets, int *index_1_offsets, int *index_0, int *index_1);

#ifdef __cplusplus
}
#endif
#endif
