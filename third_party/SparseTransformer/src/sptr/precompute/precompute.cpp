#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "precompute_cuda_kernel.h"

void precompute_all_cuda(int N, int n, const unsigned int n_max, at::Tensor counts_tensor, at::Tensor offsets_tensor, at::Tensor sq_offsets_tensor, at::Tensor index_0_offsets_tensor, at::Tensor index_1_offsets_tensor, at::Tensor index_0_tensor, at::Tensor index_1_tensor)
{
    const int *counts = counts_tensor.data_ptr<int>();
    const int *offsets = offsets_tensor.data_ptr<int>();
    const int *sq_offsets = sq_offsets_tensor.data_ptr<int>();
    int *index_0_offsets = index_0_offsets_tensor.data_ptr<int>();
    int *index_1_offsets = index_1_offsets_tensor.data_ptr<int>();
    int *index_0 = index_0_tensor.data_ptr<int>();
    int *index_1 = index_1_tensor.data_ptr<int>();
    precompute_all_cuda_launcher(N, n, n_max, counts, offsets, sq_offsets, index_0_offsets, index_1_offsets, index_0, index_1);
}
