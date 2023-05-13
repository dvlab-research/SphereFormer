#ifndef _ATTENTION_CUDA_KERNEL
#define _ATTENTION_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void attention_step1_forward_cuda(int N_q, int N_k, int M, int h, int hdim, const unsigned int n_max, at::Tensor q_tensor, at::Tensor k_tensor, at::Tensor index0_tensor, at::Tensor index1_tensor, at::Tensor attn_tensor);
void attention_step1_backward_cuda(int N, int M, int h, int hdim, const unsigned int n_max, at::Tensor grad_out_tensor, at::Tensor index0_tensor, at::Tensor index0_tensor_offsets, at::Tensor index1_tensor, at::Tensor index1_tensor_offsets, at::Tensor q_tensor, at::Tensor k_tensor, at::Tensor grad_q_tensor, at::Tensor grad_k_tensor);

void attention_step2_forward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor output_tensor);
void attention_step2_backward_cuda(int N, int M, int h, int hdim, int n_max, at::Tensor grad_out_tensor, at::Tensor index0_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor index1_offsets_tensor, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void attention_step1_forward_cuda_launcher(int N_q, int N_k, int M, int h, int hdim, const unsigned int n_max, const float *q, const float *k, const int *index0, const int *index1, float *attn);
void attention_step1_backward_cuda_launcher(int N, int M, int h, int hdim, const unsigned int n_max, const float *grad_out, const int *index0, const int *index0_offsets, const int *index1, const int *index1_offsets, const float *q, const float *k, float *grad_q, float *grad_k);

void attention_step2_forward_cuda_launcher(int N, int M, const int h, int hdim, int n_max, const float *attn, const float *v, const int *index0_offsets, const int *index1, float *output);
void attention_step2_backward_cuda_launcher(int N, int M, int h, int hdim, int n_max, const float *grad_out, const int *index0, const int *index0_offsets, const int *index1, const int *index1_offsets, const float *attn, const float *v, float *grad_attn, float *grad_v);

#ifdef __cplusplus
}
#endif
#endif
