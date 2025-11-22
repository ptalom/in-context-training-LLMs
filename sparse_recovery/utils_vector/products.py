"""
Tensor operations for Kronecker, Khatri-Rao and face-splitting products.
"""

import torch 
import numpy as np


########################################################################################
########################################################################################

def kron_batch(A, B):
    """
    Adapted from https://gist.github.com/yulkang/4a597bcc5e9ccf8c7291f8ecb776382d
    A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk

    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type A: torch.Tensor (batch_size, m, n)
    :type A: torch.Tensor (batch_size, p, q)
    :rtype: torch.Tensor (batch_size, m*p, n*q)
    """
    siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(B.shape[-2:])) # (mp, nq)
    #siz1 = torch.Size([A.shape[-2] * B.shape[-2], A.shape[-1] * B.shape[-1]]) # (mp, nq)

    res = A.unsqueeze(-1).unsqueeze(-3) * B.unsqueeze(-2).unsqueeze(-4) # (batch_size, m, p, n, q)
    siz0 = res.shape[:-4] # (batch_size)
    return res.reshape(siz0 + siz1) # (batch_size, mp, nq)

def khatri_rao_product(A, B):
    """
    Compute the Khatri-Rao product of two matrices A(m, n) and B(p, n), 
    (A ⋆ B)_{:,i} = A_{:,i} ⊗ B_{:,i}, where ⊗ is the Kronecker product.
    We have A ⋆ B = (A ⊗ 1_p) * (1_m ⊗ B), where 1_m is a m-dimensional vector of ones. 
    
    Parameters:
    A: torch.Tensor of shape (m, n)
    B: torch.Tensor of shape (p, n)
    
    Returns:
    result: torch.Tensor of shape (m*p, n)
    """
    assert A.shape[1] == B.shape[1]
    #return torch.stack([torch.kron(A[:, i], B[:, i]) for i in range(A.shape[1])], dim=1) # (mp, n)
    return torch.kron(A, torch.ones(B.shape[0], 1).to(A.device)) * torch.kron(torch.ones(A.shape[0], 1).to(A.device), B) # (mp, n)

def face_splitting_product(A, B):
    """
    Compute the face-splitting product of two tensors A(m, n) and B(m, p), 
    (A • B)_{i} = A_{i} ⊗ B_{i}, where ⊗ is the Kronecker product.
    We have A • B = (A ⊗ 1_p^T) * (1_n^T ⊗ B), where 1_n is a n-dimensional vector of ones. 

    Parameters:
    A: torch.Tensor of shape (m, n)
    B: torch.Tensor of shape (m, p)

    Returns:
    result: torch.Tensor of shape (m, n*p) such that the line i is the kronecker product of the line i of A and B.
    """
    assert A.shape[0] == B.shape[0]
    #return torch.stack([torch.kron(A[i], B[i]) for i in range(A.shape[0])], dim=0) # (m, np)
    return torch.kron(A, torch.ones(1, B.shape[1]).to(A.device)) * torch.kron(torch.ones(1, A.shape[1]).to(A.device), B)  # (m, np)

## Numpy version

def khatri_rao_product_numpy(A, B):
    """
    Compute the Khatri-Rao product of two matrices A(m, n) and B(p, n), 
    (A ⋆ B)_{:,i} = A_{:,i} ⊗ B_{:,i}, where ⊗ is the Kronecker product.
    We have A ⋆ B = (A ⊗ 1_p) * (1_m ⊗ B), where 1_m is a m-dimensional vector of ones. 
    
    Parameters:
    A: np.array of shape (m, n)
    B: np.array of shape (p, n)
    
    Returns:
    result: np.array of shape (m*p, n)
    """
    assert A.shape[1] == B.shape[1]
    #return np.stack([np.kron(A[:, i], B[:, i]) for i in range(A.shape[1])], axis=1) # (mp, n)
    return np.kron(A, np.ones((B.shape[0], 1))) * np.kron(np.ones((A.shape[0], 1)), B) # (mp, n)

def face_splitting_product_numpy(A, B):
    """
    Compute the face-splitting product of two tensors A(m, n) and B(m, p), 
    (A • B)_{i} = A_{i} ⊗ B_{i}, where ⊗ is the Kronecker product.
    We have A • B = (A ⊗ 1_p^T) * (1_n^T ⊗ B), where 1_n is a n-dimensional vector of ones. 

    Parameters:
    A: np.array of shape (m, n)
    B: np.array of shape (m, p)

    Returns:
    result: np.array of shape (m, n*p) such that the line i is the kronecker product of the line i of A and B.
    """
    assert A.shape[0] == B.shape[0]
    #return np.stack([np.kron(A[i], B[i]) for i in range(A.shape[0])], axis=0) # (m, np)
    return np.kron(A, np.ones((1, B.shape[1]))) * np.kron(np.ones((1, A.shape[1])), B)  # (m, np)

########################################################################################
########################################################################################

if __name__ == "__main__":

    m, n, p, q = 2, 3, 4, 5
    batch_size = 10
    A = torch.randn(batch_size, m, n)
    B = torch.randn(batch_size, p, q)
    C1 = kron_batch(A, B)
    C2 = torch.stack([torch.kron(A[i], B[i]) for i in range(batch_size)], dim=0)
    print(torch.allclose(C1, C2))

    m, n, p = 2, 3, 4
    A = torch.randn(m, n)
    B = torch.randn(p, n)
    C1 = khatri_rao_product(A, B)
    C2 = torch.stack([torch.kron(A[:, i], B[:, i]) for i in range(n)], dim=1)
    print(torch.allclose(C1, C2))

    m, n, p = 2, 3, 4
    A = torch.randn(m, n)
    B = torch.randn(m, p)
    C1 = face_splitting_product(A, B)
    C2 = torch.stack([torch.kron(A[i], B[i]) for i in range(m)], dim=0)
    print(torch.allclose(C1, C2))