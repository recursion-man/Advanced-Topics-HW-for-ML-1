#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MATRIX_SIZE 32
#define TILE_SIZE 32

template <typename scalar_t>
__global__ void matmul_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    int num_matrices
) {
    int batch_idx = blockIdx.x;

    if (batch_idx >= num_matrices) return;

    // Adjust pointers for the current batch
    a += batch_idx * MATRIX_SIZE * MATRIX_SIZE;
    b += batch_idx * MATRIX_SIZE * MATRIX_SIZE;
    out += batch_idx * MATRIX_SIZE * MATRIX_SIZE;

    __shared__ scalar_t a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t b_tile[TILE_SIZE][TILE_SIZE];

    int row = threadIdx.y;
    int col = threadIdx.x;

    a_tile[row][col] = a[row * MATRIX_SIZE + col];
    b_tile[row][col] = b[row * MATRIX_SIZE + col];

    __syncthreads();

    scalar_t value = 0;

    #pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
        value += a_tile[row][i] * b_tile[i][col];
    }

    __syncthreads();

    out[row * MATRIX_SIZE + col] = value;
}

at::Tensor matmul_cuda(at::Tensor a, at::Tensor b, int num_matrices) {
    auto out = torch::zeros({num_matrices, MATRIX_SIZE, MATRIX_SIZE}, a.options());

    const dim3 block_dim(TILE_SIZE, TILE_SIZE);
    const dim3 grid_dim(num_matrices);  // One block per matrix pair

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "matmul_kernel", ([&] {
        matmul_kernel<scalar_t><<<grid_dim, block_dim>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            num_matrices
        );
    }));

    return out;
}
