#include <torch/extension.h>

#define MATRIX_SIZE 32

template <typename scalar_t>
__global__ void matmul_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out,
) {

    // Remember the block is quare so blockDim.x = blockDim.y
    const int blocksize = blockDim.x 


    // Shared memory defined when the kernel was called. 
    // Accessing it using __shared__
    extern __shared__ scalar_t shared_mem[]; // 

    // Split shared memory euqally between a_sub and b_sub
    scalar_t* a_sub_base = shared_mem;
    scalar_t* b_sub_base = shared_mem + blocksize * blocksize;

    // Accessors
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> a_sub(
        a_sub_base, {blocksize, blocksize});
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b_sub(
        b_sub_base, {blocksize, blocksize});

    // Global id (x,y) in the matrix
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    

    if (x >= out.size(0) || y >= out.size(1)) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bpg = gridDim.x;  


    scalar_t tmp = 0.0;
    for (int i = 0; i < bpg; ++i) {

        int b_sub_row = tx + i * blocksize;
        int a_sub_col = ty + i * blocksize;

        if (b_sub_row < b.size(0) && a_sub_col < a.size(1)) {
            // Load data into shared memory
            a_sub[tx][ty] = a[x][a_sub_col];
            b_sub[tx][ty] = b[b_sub_row][y];
        } else {
            a_sub[tx][ty] = 0.0;
            b_sub[tx][ty] = 0.0;
        }


        __syncthreads();


        for (int j = 0; j < blocksize; ++j) {
            tmp += a_sub[tx][j] * b_sub[j][ty];
        }

        __syncthreads();
    }


    out[x][y] = tmp;
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    auto out = torch::zeros({MATRIX_SIZE, MATRIX_SIZE}, a.options());
    const int blocksize  = at::cuda::getCurrentDeviceProperties()->warpSize;
    const dim3 block_dim(blocksize , blocksize );
    const int grid_size  = (MATRIX_SIZE + blocksize  - 1) / blocksize 
    const dim3 grid_dim(grid_size , grid_size);
    const int shared_mem_size = 2 * blocksize * blocksize * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "matmul_kernel", ([&] {
        matmul_kernel<scalar_t><<<grid_dim, block_dim, shared_mem_size>>>(
            a.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        );
    }));

    return out;
}
