import torch
import time
from matmul import matmul

def check_correctness(a1_ls, a2_ls):
    results = matmul(a1_ls, a2_ls)
    for a1, a2, result in zip(a1_ls, a2_ls, results):
        expected = torch.matmul(a1, a2)
        if not torch.allclose(result.cpu(), expected.cpu(), atol=1e-6):
            print("Test failed.")
            return False
    print("All tests passed.")
    return True

if __name__ == "__main__":
    # Generate example tensors
    num_matrices = 10000

    a1_ls = [torch.rand((32, 32), dtype=torch.float32) for _ in range(num_matrices)]
    a2_ls = [torch.rand((32, 32), dtype=torch.float32) for _ in range(num_matrices)]


    # Check correctness
    correctness = check_correctness(a1_ls, a2_ls)

