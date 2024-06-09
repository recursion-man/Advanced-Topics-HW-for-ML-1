import torch
import time

def generate_matrices(num_matrices, size):
    return [torch.rand(size, dtype=torch.float32) for _ in range(num_matrices)]

def benchmark_built_in_matmul(a1_ls, a2_ls):
    start_time = time.time()
    [torch.matmul(a1.to('cuda:0'), a2.to('cuda:0')) for a1, a2 in zip(a1_ls, a2_ls)]
    end_time = time.time()
    
    return end_time - start_time

if __name__ == "__main__":
    # Set parameters
    num_matrices = 100000
    matrix_size = 32

    # Generate input matrices on the CPU
    a1_ls = generate_matrices(num_matrices, (matrix_size, matrix_size))
    a2_ls = generate_matrices(num_matrices, (matrix_size, matrix_size))

    # Benchmark built-in matmul
    built_in_time = benchmark_built_in_matmul(a1_ls, a2_ls)
    print(f"PyTorch built-in matmul time: {built_in_time:.6f} seconds")
