#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "attention_cuda_kernel.h"


void attention_step1_forward_cuda(int N_q, int N_k, int M, int h, int hdim, const unsigned int n_max, at::Tensor q_tensor, at::Tensor k_tensor, 
    at::Tensor index0_tensor, at::Tensor index1_tensor, at::Tensor attn_tensor)
{
    const float *q = q_tensor.data_ptr<float>();
    const float *k = k_tensor.data_ptr<float>();
    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    float *attn = attn_tensor.data_ptr<float>();
    attention_step1_forward_cuda_launcher(N_q, N_k, M, h, hdim, n_max, q, k, index0, index1, attn);
}

void attention_step1_backward_cuda(int N, int M, int h, int hdim, const unsigned int n_max, at::Tensor grad_out_tensor, 
    at::Tensor index0_tensor, at::Tensor index0_tensor_offsets, at::Tensor index1_tensor, at::Tensor index1_tensor_offsets, at::Tensor q_tensor, at::Tensor k_tensor, 
    at::Tensor grad_q_tensor, at::Tensor grad_k_tensor)
{
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index0_offsets = index0_tensor_offsets.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const int *index1_offsets = index1_tensor_offsets.data_ptr<int>();
    const float *q = q_tensor.data_ptr<float>();
    const float *k = k_tensor.data_ptr<float>();
    float *grad_q = grad_q_tensor.data_ptr<float>();
    float *grad_k = grad_k_tensor.data_ptr<float>();
    attention_step1_backward_cuda_launcher(N, M, h, hdim, n_max, grad_out, index0, index0_offsets, index1, index1_offsets, q, k, grad_q, grad_k);
}

void attention_step2_forward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor attn_tensor, at::Tensor v_tensor, 
    at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor output_tensor)
{
    const float *attn = attn_tensor.data_ptr<float>();
    const float *v = v_tensor.data_ptr<float>();
    const int *index0_offsets = index0_offsets_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    float *output = output_tensor.data_ptr<float>();
    attention_step2_forward_cuda_launcher(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, output);
}

void attention_step2_backward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor grad_out_tensor, at::Tensor index0_tensor,
    at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor index1_offsets_tensor, at::Tensor attn_tensor, at::Tensor v_tensor, 
    at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor)
{
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index0_offsets = index0_offsets_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const int *index1_offsets = index1_offsets_tensor.data_ptr<int>();
    const float *attn = attn_tensor.data_ptr<float>();
    const float *v = v_tensor.data_ptr<float>();
    float *grad_attn = grad_attn_tensor.data_ptr<float>();
    float *grad_v = grad_v_tensor.data_ptr<float>();
    attention_step2_backward_cuda_launcher(N, M, h, hdim, n_max, grad_out, index0, index0_offsets, index1, index1_offsets, attn, v, grad_attn, grad_v);
}
