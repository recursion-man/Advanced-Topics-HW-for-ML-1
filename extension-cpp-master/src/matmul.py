import matmul_cuda
import torch

def matmul(a1_ls, a2_ls):
    # Stack the list of tensors into a single batch tensor
    a1_batch = torch.stack(a1_ls).to('cuda:0')
    a2_batch = torch.stack(a2_ls).to('cuda:0')
    
    # Perform batched matrix multiplication
    out_batch = matmul_cuda.matmul(a1_batch, a2_batch)
    
    # Convert the batch tensor back into a list of tensors
    out_ls = [out_batch[i] for i in range(out_batch.size(0))]
    
    return out_ls
