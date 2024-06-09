# src/matmul.py

import matmul_cuda


def matmul(a1_ls, a2_ls, cuda_device='cuda:0'):
    out_ls = []
    
    for a1, a2 in zip(a1_ls, a2_ls):
        a1 = a1.to(cuda_device)
        a2 = a2.to(cuda_device)
        out = matmul_cuda.matmul(a1, a2)
        out_ls.append(out)
    
    return out_ls
