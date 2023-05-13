#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "attention/attention_cuda_kernel.h"
#include "rpe/relative_pos_encoding_cuda_kernel.h"
#include "precompute/precompute_cuda_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_step1_forward_cuda", &attention_step1_forward_cuda, "attention_step1_forward_cuda");
    m.def("attention_step1_backward_cuda", &attention_step1_backward_cuda, "attention_step1_backward_cuda");
    m.def("attention_step2_forward_cuda", &attention_step2_forward_cuda, "attention_step2_forward_cuda");
    m.def("attention_step2_backward_cuda", &attention_step2_backward_cuda, "attention_step2_backward_cuda");
    m.def("precompute_all_cuda", &precompute_all_cuda, "precompute_all_cuda");
    m.def("dot_prod_with_idx_forward_cuda", &dot_prod_with_idx_forward_cuda, "dot_prod_with_idx_forward_cuda");
    m.def("dot_prod_with_idx_backward_cuda", &dot_prod_with_idx_backward_cuda, "dot_prod_with_idx_backward_cuda");
    m.def("attention_step2_with_rel_pos_value_forward_cuda", &attention_step2_with_rel_pos_value_forward_cuda, "attention_step2_with_rel_pos_value_forward_cuda");
    m.def("attention_step2_with_rel_pos_value_backward_cuda", &attention_step2_with_rel_pos_value_backward_cuda, "attention_step2_with_rel_pos_value_backward_cuda");
    m.def("dot_prod_with_idx_all_forward_cuda", &dot_prod_with_idx_all_forward_cuda, "dot_prod_with_idx_all_forward_cuda");
}
