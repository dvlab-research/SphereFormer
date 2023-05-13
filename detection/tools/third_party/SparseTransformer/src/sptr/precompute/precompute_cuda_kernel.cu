#include "../cuda_utils.h"
#include "precompute_cuda_kernel.h"

__global__ void precompute_all_cuda_kernel( // M, h, C//h
    int N, int n, int k, const int *counts, const int *offsets, const int *sq_offsets, int *index0_offsets, int *index1_offsets, int *index0, int *index1) {
    // counts: (n), sq_offsets: (n), index0_offsets: (n), index1_offsets: (n)

    int n_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    int start = offsets[n_idx];
    int start_val = sq_offsets[n_idx];
    int length = counts[n_idx];
    for(int t_idx = thread_idx; t_idx < length; t_idx += blockDim.x){
        index0_offsets[start+t_idx] = start_val + length * t_idx;
        index1_offsets[start+t_idx] = start_val + t_idx;
        for(int i = 0; i < length; i++){
            index0[start_val + i*length + t_idx] = start+i;
            index1[start_val + i*length + t_idx] = start+t_idx;
        }
    }
}

void precompute_all_cuda_launcher(int N, int n, const unsigned int n_max, const int *counts, 
    const int *offsets, const int *sq_offsets, int *index_0_offsets, int *index_1_offsets, int *index_0, int *index_1) {
    // input: attn: (M, h), index0: (M, ), index1: (M, )

    unsigned int blocks = n;
	unsigned int n_threads = opt_n_threads(n_max);
    n_threads = n_threads == n_max ? n_threads : n_threads * 2;
    n_threads = n_threads > 1024 ? 1024 : n_threads;

    precompute_all_cuda_kernel<<<blocks, n_threads, 0>>>(N, n, n_max, counts, offsets, sq_offsets, index_0_offsets, index_1_offsets, index_0, index_1);

}