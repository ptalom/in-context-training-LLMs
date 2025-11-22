"""
This module provides functions for generating random index uplets for matrix and tensor completion tasks.
It includes functions for both matrix and tensor completion, allowing for one-hot encoding of indices and 
reproducibility through a random seed.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math

########################################################################################
########################################################################################
######### Get matrices of a specific rank

def get_matrices(
    d_1, d_2, rank, batch_size=1, symmetric=False, normalize=True, scale=None, 
    seed=None, generator:torch.Generator=None, verbose=False):
    """
    Generate a batch of matrices of size (d_1, d_2) with a rank equal to 'rank'
    d_1 : int
        First dimension of the matrix
    d_2 : int
        Second dimension of the matrix
    rank : int
        Rank of the matrix
    batch_size : int
        Number of matrices to generate
    symmetric : bool
        If True, the matrices are symmetric
        If symmetric is True, d_1 must be equal to d_2
    normalize : bool
        If True, the matrices are normalized such that the Frobenius norm is equal to sqrt(d_1*d_2)
    scale : float
        Scale factor to apply to the matrices, if None, scale = 1/sqrt(rank)
        The singular values of the matrices are approximately equal to 'scale'
    seed : int
        Seed for the random number generator
    generator : torch.Generator
        Generator for the random number generator
    verbose : bool
        If True, print the singular values of the matrices

    Return : W ~ (batch_size, d_1, d_2)
    Adapted from : https://github.com/roosephu/deep_matrix_factorization/blob/master/gen_gt.py#L16C1-L29C1
    """
    
    if seed :
        generator = torch.Generator()
        generator.manual_seed(seed)

    assert rank <= min(d_1, d_2), "rank must be less than or equal to min(d_1, d_2)"
    assert not symmetric or d_1==d_2, "d_1 must be equal to d_2 if symmetric is True"
    U = torch.randn(batch_size, d_1, rank, generator=generator) # (batch_size, d_1, rank)
    if symmetric:
        V = U # (batch_size, d_1, rank)
    else:
        V = torch.randn(batch_size, d_2, rank, generator=generator) # (batch_size, d_2, rank)
        
    scaler = 1/math.sqrt(rank) if scale is None else scale
    W = torch.bmm(U, V.transpose(1, 2)) * scaler # (batch_size, d_1, rank) x (batch_size, rank, d_2) = (batch_size, d_1, d_2)
    if normalize :
        # compute W[b] / frobenius_norm(W[b]) for each b
        W = W / torch.norm(W, dim=(1, 2), p='fro', keepdim=True) * np.sqrt(d_1 * d_2) # (batch_size, d_1, d_2)

    U, Sigma, VT = torch.linalg.svd(W) # (batch_size, d_1, d_1), (batch_size, min(d_1, d_2)), (batch_size, d_2, d_2)
    V = VT.transpose(1, 2) # (batch_size, d_2, d_2)

    if verbose:
        print(f"singular values = {Sigma[:,:rank]}, Fro(w) = {torch.norm(W, dim=(1, 2), p='fro')}")
        
    return (W.squeeze(0), U.squeeze(0), Sigma.squeeze(0), V.squeeze(0))  if batch_size==1 else (W, U, Sigma, V)

########################################################################################
########################################################################################

def data_matrix_completion(A, one_hot=False, seed=None):
    """

    This function generates random index pairs for a 2D tensor (matrix) A 
    of shape (n, n) and retrieves the values at those indices. 
    The indices can optionally be returned as one-hot encoded vectors.

    Parameters:
    - A (torch.Tensor): The input matrix of shape (n, n).
    - one_hot (bool): If True, returns one-hot encoded representations of the indices.
    - seed (int, optional): Random seed for reproducibility. If provided, the random state is restored after execution.

    Returns:
    - XY (torch.Tensor): A tensor of shape (n^2, 2, n) containing the random index pairs. 
    - Z (torch.Tensor): A tensor of shape (n^2, 1) containing the values from matrix A 
      at the generated indices, where each value corresponds to the calculation: 
      Z_{ij} = X_i^T A Y_i = A_{ij}.

    Note:
    - The output indices are sampled uniformly at random from the matrix A.
    """
    if seed is not None :
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
    
    ############################################

    n = A.shape[0]
    # ## Generate all possible indices (stochastic permutation)
    # perm = torch.randperm(n**2)
    # x, y = perm // n, perm % n # (n^2,), (n^2,)
    ## Generate all possible indices using meshgrid (deteministic permutation)
    indices = torch.arange(n) # Indices from 0 to n-1
    x = indices.repeat_interleave(n) # (n^2,)
    y = indices.repeat(n) # (n^2,)

    XY = torch.stack([x, y], dim=1) # (n^2, 2)
    if one_hot : XY = F.one_hot(XY, num_classes=n).float() # (n^2, 2, n)
    Z = A[x, y] # (n^2,)
    Z = Z.unsqueeze(1) # (n^2, 1)

    ############################################

    # n_1, n_2 = A.shape
    # # ## Generate all possible indices (stochastic permutation)
    # # perm = torch.randperm(n_1*n_2)
    # # x, y = (perm // n_2) % n_1, perm % n_2 # (n_1*n_2,), (n_1*n_2,)
    # ## Generate all possible indices using meshgrid (deteministic permutation)
    # x = torch.arange(n_1).repeat_interleave(n_2)  # (n_1*n_2,)
    # y = torch.arange(n_2).repeat(n_1)  # (n_1*n_2,)
    
    # XY = [x, y] # (2, n_1*n_2)
    # if one_hot : 
    #     XY[0] = F.one_hot(x, num_classes=n_1).float() # (n^2*n_2, n_1)
    #     XY[1] = F.one_hot(y, num_classes=n_2).float() # (n^2*n_2, n_2)
    # Z = A[x, y] # (n_1*n_2,)
    # Z = Z.unsqueeze(1) # (n_1*n_2, 1)

    ############################################

    if seed is not None : torch.set_rng_state(old_state)

    return XY, Z

########################################################################################
########################################################################################

def data_tensor_completion(A, one_hot=False, seed=None):
    """
    This function generates random indices for a R-order tensor A of shape 
    (n, n, ..., n) and retrieves the values at those indices. 
    The indices can optionally be returned as one-hot encoded vectors.

    Parameters:
    - A (torch.Tensor): The input tensor of shape (n, ..., n) of order R.
    - one_hot (bool): If True, returns one-hot encoded representations of the indices for each dimension.
    - seed (int, optional): Random seed for reproducibility. If provided, the random state is restored after execution.

    Returns:
    - X (list of torch.Tensor): A list containing the random indices for each dimension. 
      Each tensor in the list has shape (n^R,), in the non one-hot encoded case, and (n^R, n) in the one-hot encoded case.
    - y (torch.Tensor): A tensor of shape (n^R, 1) containing the values from tensor A 
      at the generated indices, calculated using the formula:
      Z_i = ∑ A[j_1, ..., j_R] * X^{(1)}_i[j_1] * X^{(2)}_i[j_2] * ... * X^{(R)}_i[j_R].
    """
    # if A.dim()==2:
    #     return data_matrix_completion(A, one_hot, seed=seed)

    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)

    ############################################
    shape = A.shape
    R = len(shape)  # Number of dimensions
    n = shape[0]  # Dimension of the tensor

    # ## Generate all possible indices (stochastic permutation)
    # perm = torch.randperm(n**R) # (n^R,)
    # indices = [(perm // (n**i)) % n for i in range(R)] # (R, n^R)
    ## Generate all possible indices using meshgrid (deteministic permutation)
    grids = torch.meshgrid(*[torch.arange(s) for s in shape], indexing='ij') # (R, n, ..., n)
    indices = [grid.flatten() for grid in grids]  # (R, n^R)

    # Stack the indices 
    X = torch.stack(indices, dim=1) # (n^R, n)
    # One-hot encoding if required
    if one_hot : X = F.one_hot(X, num_classes=n).float() # (n^R, R, n)
    # Use the generated indices to extract the corresponding values from the tensor M
    Z = A[tuple(indices)]  # Advanced indexing over multiple dimensions
    # Ensure Z has shape (n^R, 1)
    Z = Z.unsqueeze(1)

    ############################################

    if seed is not None:
        torch.set_rng_state(old_state)

    return X, Z  # Returning (X^{(1)}, ..., X^{(R)}), and Z

########################################################################################
########################################################################################

def data_tensor_completion_general(A, one_hot=False, seed=None):
    """
    Generalized data tensor completion for tensors of any order R with different dimensions.

    This function generates random indices for a tensor A of shape (n_1, n_2, ..., n_R) 
    and computes the corresponding measurement values based on those indices.

    Parameters:
    - A (torch.Tensor): The input tensor of shape (n_1, n_2, ..., n_R).
    - one_hot (bool): If True, returns one-hot encoded representations of the indices for each dimension.
    - seed (int, optional): Random seed for reproducibility. If provided, the random state is restored after execution.

    Returns:
    - X (list of torch.Tensor): A list containing the random indices for each dimension, X^{(1)}, ..., X^{(R)}. 
      In the non ohe-ont case, each tensor in the list has shape (N,), where N is the total number of combinations
      In the one-hot encoded case, each tensor has shape (N, n_i) for the i-th dimension.
    - Z (torch.Tensor): A tensor of shape (N, 1) containing the computed values based on the tensor A.
      Each element Z[i] is calculated from the selected indices according to the formula:
      Z_i = ∑ A[j_1, ..., j_R] * X^{(1)}_i[j_1] * X^{(2)}_i[j_2] * ... * X^{(R)}_i[j_R].

    Note:
    - The total number of measurements N is the product of the dimensions of A: N = n_1 * n_2 * ... * n_R.
    """

    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)

    # Get the dimensions of the tensor
    shape = A.shape
    R = len(shape)  # Number of dimensions
    N = torch.prod(torch.tensor(shape)).item()  # Total number of combinations




    # ## Generate all possible indices (stochastic permutation)
    perm = torch.randperm(N) # (N,)
    indices = [(torch.div(perm, torch.prod(torch.tensor(shape[r+1:])) if r+1 < R else 1)) % shape[r] for r in range(R)] # (R, N)
    X = [indices[r] for r in range(R)] # (R, N)
    ## Generate all possible indices using meshgrid (deteministic permutation)
    grids = torch.meshgrid(*[torch.arange(dim) for dim in shape], indexing='ij')
    indices = [grid.flatten() for grid in grids]  # Flatten each grid to create combinations
    X = [index for index in indices]  # List of index tensors for each dimension

    # Create a one-hot encoding if required
    if one_hot:
        X = [F.one_hot(x, num_classes=shape[r]).float() for r, x in enumerate(X)]  # One-hot encoding for each dimension

    # Use the generated indices to extract the corresponding values from the tensor A
    Z = A[tuple(indices)]  # This gets the values according to the randomly generated indices
    Z = Z.unsqueeze(1) # (N, 1)

    if seed is not None:
        torch.set_rng_state(old_state)

    return X, Z  # Returning (X^{(1)}, ..., X^{(R)}), and Z

########################################################################################
########################################################################################



    