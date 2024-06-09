#include <torch/extension.h>
#include <vector>

torch::Tensor matmul_cuda(torch::Tensor a1, torch::Tensor a2, int num_matrices);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor matmul(torch::Tensor a1, torch::Tensor a2) {
    CHECK_INPUT(a1);
    CHECK_INPUT(a2);
    int num_matrices = a1.size(0);
    return matmul_cuda(a1, a2, num_matrices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "Matrix Multiplication (CUDA)");
}
