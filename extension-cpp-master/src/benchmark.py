import torch
import time
import matmul_cuda

def generate_matrices(num_matrices, size):
    return [torch.rand(size, dtype=torch.float32) for _ in range(num_matrices)]

def benchmark_custom_matmul(a1_ls, a2_ls):
    start_time = time.time()
    a1_batch = torch.stack(a1_ls).to('cuda:0')
    a2_batch = torch.stack(a2_ls).to('cuda:0')
    out_batch = matmul_cuda.matmul(a1_batch, a2_batch).to('cpu')
    out_ls = [out_batch[i] for i in range(out_batch.size(0))]
    end_time = time.time()
    return (end_time - start_time) * 1e3  

def benchmark_pytorch_matmul(a1_ls, a2_ls):
    start_time = time.time()
    a1_batch = torch.stack(a1_ls).to('cuda:0')
    a2_batch = torch.stack(a2_ls).to('cuda:0')
    out_batch = torch.bmm(a1_batch, a2_batch).to('cpu')
    out_ls = [out_batch[i] for i in range(out_batch.size(0))]
    end_time = time.time()
    return (end_time - start_time) * 1e3  

def run_benchmarks(benchmark_func, num_matrices, matrix_size, iterations, label):
    for _ in range(2):
        a1_ls = generate_matrices(num_matrices, (matrix_size, matrix_size))
        a2_ls = generate_matrices(num_matrices, (matrix_size, matrix_size))
        benchmark_func(a1_ls, a2_ls)

    times = []
    for _ in range(iterations):
        a1_ls = generate_matrices(num_matrices, (matrix_size, matrix_size))
        a2_ls = generate_matrices(num_matrices, (matrix_size, matrix_size))
        time_taken = benchmark_func(a1_ls, a2_ls)
        times.append(time_taken)
    avg_time = sum(times) / iterations
    print(f"{label} average time over {iterations} iterations; {num_matrices} matrices; { avg_time:.2f} milliseconds")
   
    

if __name__ == "__main__":
    num_matrices = 100000
    matrix_size = 32
    iterations = 40



 # Run PyTorch matmul benchmarks
    run_benchmarks(benchmark_pytorch_matmul, num_matrices, matrix_size, iterations, "PyTorch matmul")

    # Run custom matmul benchmarks
    run_benchmarks(benchmark_custom_matmul, num_matrices, matrix_size, iterations, "Custom matmul")
    
   

