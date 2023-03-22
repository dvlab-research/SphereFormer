#include "../cuda_utils.h"
#include "attention_cuda_kernel.h"

__global__ void attention_step1_forward_cuda_kernel( // M, h, C//h
    int N_q, int N_k, int M, int h, int d, const float *q, const float *k,
    const int *index0, const int *index1, float *attn) {
    // q: [N, h, d], k: [h, d, N], index0: [M], index1: [M], attn: [h, M]

    int h_idx = blockIdx.y;
    int m_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(m_idx >= M) return;
    float s = 0.0;
    int index_q = index0[m_idx], index_k = index1[m_idx];
    for(int i = 0; i < d; i++){
        s += q[h_idx * d * N_q + i * N_q + index_q] * k[h_idx * d * N_k + i * N_k + index_k];
    }
    attn[h_idx * M + m_idx] = s;
}

void attention_step1_forward_cuda_launcher(int N_q, int N_k, int M, int h, int hdim, const unsigned int n_max, 
    const float *q, const float *k, const int *index0, const int *index1, float *attn) {
    // input: attn: (h, M), index0: (M, ), index1: (M, )
    unsigned int n_threads = 512;
    dim3 blocks((M + n_threads - 1) / n_threads, h);
    attention_step1_forward_cuda_kernel<<<blocks, n_threads, 0>>>(N_q, N_k, M, h, hdim, q, k, index0, index1, attn);
}

__global__ void attention_step1_backward_cuda_kernel( // M, h, C//h
    int N, int M, int h, int d, const float *grad_out, const int *index0, const int *index0_offsets, const int *index1, const int *index1_offsets, 
    const float *q, const float *k, float *grad_q, float *grad_k) {
    // q: [N, h, d], k: [N, h, d], index0: [M], index1: [M], attn: [M, h], grad_out: [M, h]
    // grad_q: [N, h, hdim], grad_k: [N, h, hdim]

    int n_h = blockDim.x;
    int h_idx = blockIdx.y * n_h + threadIdx.y;
    int q_idx = blockIdx.x;
    int d_idx = threadIdx.x;
    int C = d * h;
    
    int start = index0_offsets[q_idx], end = index0_offsets[q_idx+1];
    int n = end - start;

    float grad_q_val = 0;
    for(int i = 0; i < n; i++){
        int start_i = start + i;
        float grad_out_val = grad_out[start_i*h + h_idx];
        int k_idx = index1[start_i];
        grad_q_val += grad_out_val * k[k_idx*C + h_idx*d + d_idx];
    }
    grad_q[q_idx*C + h_idx*d + d_idx] = grad_q_val;

    float grad_k_val = 0;
    int start_k = index1_offsets[q_idx];
    for(int i = 0; i < n; i++){
        int start_i = start_k + i*n;
        float grad_out_val = grad_out[start_i*h + h_idx];
        int query_idx = index0[start_i];
        grad_k_val += grad_out_val * q[query_idx*C + h_idx*d + d_idx];
    }
    grad_k[q_idx*C + h_idx*d + d_idx] = grad_k_val;
}

void attention_step1_backward_cuda_launcher(int N, int M, int h, int hdim, const unsigned int n_max, 
    const float *grad_out, const int *index0, const int *index0_offsets, const int *index1, const int *index1_offsets, const float *q, const float *k, float *grad_q, float *grad_k) {  
    // input: grad_output: (n, nsample, c), output: grad_input1: (n, c), grad_input2: (n, c)

	unsigned int n_h = h*hdim > 512 ? 512 / hdim : h;

    dim3 blocks(N, h/n_h);
    dim3 threads(hdim, n_h);

    attention_step1_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, grad_out, index0, index0_offsets, index1, index1_offsets, q, k, grad_q, grad_k);
    
}

__global__ void attention_step2_forward_cuda_kernel( // M, h, hdim
    int N, int M, const int h, int d, const float *attn, const float *v,
    const int *index0_offsets, const int *index1, float *output) {
    // input: attn: (M, h), v: (N, h, hdim), index0: (M, ), index1: (M, ), table: (L, 3, h, hdim), rel_idx: (M, 3)

    int q_idx = blockIdx.x;
    int n_h = blockDim.x;
    int h_idx = blockIdx.y * n_h + threadIdx.y;
    int d_idx = threadIdx.x;

    int C = h*d;

    int start = index0_offsets[q_idx], end = index0_offsets[q_idx+1];
    int n = end - start;
    float sum = 0;
    for(int i = 0; i < n; i++){
        int start_i = start + i;
        int k_idx = index1[start_i];
        float v_val = v[k_idx*C + h_idx*d + d_idx];
        sum += attn[start_i*h + h_idx] * v_val;
    }
    output[q_idx*C + h_idx*d + d_idx] = sum;
}

void attention_step2_forward_cuda_launcher(int N, int M, const int h, int hdim, int n_max, const float *attn, const float *v, const int *index0_offsets,
    const int *index1, float *output) {
    // input: attn: (M, h), v: (N, h, hdim), index0: (M, ), index1: (M, ), table: (L, h, hdim, 3), rel_idx: (M, 3)
    unsigned int n_h = h*hdim > 512 ? 512 / hdim : h;
    dim3 blocks(N, h/n_h);
    dim3 threads(hdim, n_h);
    attention_step2_forward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, attn, v, index0_offsets, index1, output);
}

__global__ void attention_step2_grad_v_backward_cuda_kernel( // M, h, hdim
    int N, int M, int h, int hdim, const float *grad_out, const int *index0, const int *index0_offsets, const int *index1, const int *index1_offsets, const float *attn, const float *v,
    float *grad_v) {
    // input: attn: (M, h), v: (h, hdim, N), index0: (M, ), index1: (M, ), rel_idx: (3, M)

    int q_idx = blockIdx.x;
    int n_h = blockDim.x;
    int h_idx = blockIdx.y * n_h + threadIdx.y;
    int d_idx = threadIdx.x;

    int C = h*hdim;

    int start = index0_offsets[q_idx], end = index0_offsets[q_idx+1];
    int n = end - start;
    int start_k = index1_offsets[q_idx];
    float grad_v_val = 0;
    
    for(int i = 0; i < n; i ++){
        int start_i = start_k + i*n;
        int query_idx = index0[start_i];
        float grad_out_val = grad_out[query_idx*C + h_idx*hdim + d_idx];
        
        float grad_val = attn[start_i*h + h_idx] * grad_out_val;
        grad_v_val += grad_val;
    }
    grad_v[q_idx*C + h_idx*hdim + d_idx] = grad_v_val;

}

void attention_step2_backward_cuda_launcher(int N, int M, int h, int hdim, int n_max, const float *grad_out, const int *index0, const int *index0_offsets, 
    const int *index1, const int *index1_offsets, const float *attn, const float *v, float *grad_attn, float *grad_v) {  
    // input: grad_out: (N, h, hdim)
    
    unsigned int n_h = h*hdim > 512 ? 512 / hdim : h;
    dim3 blocks(N, h/n_h);
    dim3 threads(hdim, n_h);
    attention_step2_grad_v_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, grad_out, index0, index0_offsets, index1, index1_offsets, attn, v, grad_v);

    unsigned int n_threads = 512;
    dim3 blocks_2((M + n_threads - 1) / n_threads, h);

    attention_step1_forward_cuda_kernel<<<blocks_2, n_threads, 0>>>(N, N, M, h, hdim, grad_out, v, index0, index1, grad_attn);
}
