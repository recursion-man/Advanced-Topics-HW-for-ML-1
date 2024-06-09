#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor matmul_cuda(torch::Tensor a1, torch::Tensor a2);

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor matmul(torch::Tensor a1, torch::Tensor a2) {
    CHECK_INPUT(a1);
    CHECK_INPUT(a2);
    return matmul_cuda(a1, a2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "Matrix multiplication (CUDA)");
}
