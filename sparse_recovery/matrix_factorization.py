import torch
import numpy as np
from numpy.linalg import svd
import cvxpy as cp
from tqdm import tqdm

import matplotlib.pyplot as plt

##

# # Set the working directory to the parent directory of your top-level package
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sparse_recovery.compressed_sensing import create_signal, create_orthonormal_basis, get_measures
from sparse_recovery.utils_vector.svd import np_SVD
from sparse_recovery.utils_vector.products import face_splitting_product_numpy
from sparse_recovery.utils_vector.data_tensor_completion import get_matrices, data_tensor_completion_general
########################################################################################
########################################################################################

def np_vec(A): return A.reshape(A.shape[0]*A.shape[1], order='F')
def np_unvec(x, m, n):
    """Inverse of the vectorization operator : mn ---> (m, n)
    https://math.stackexchange.com/a/3122442/1020794"""
    Im, In = np.eye(m), np.eye(n)
    return np.kron(np_vec(In).T, Im) @ np.kron(In, np.expand_dims(x, 1))

########################################################################################
########################################################################################

def create_matrix(n_1, n_2, r, distribution="normal", U=None, V=None, scaler=None, seed=0):
    """
    Generate a matrix A_star of size n_1 x n_2 with rank r

    :param n_1: int, number of rows
    :param n_2: int, number of columns
    :param r: int, rank of the matrix
    :param distribution: str, distribution of the singular values
    :param U: np.ndarray, left singular vectors
    :param V: np.ndarray, right singular vectors
    :param scaler: float, scaling factor for the singular values
    :param seed: int, random seed
    
    :return: np.ndarray, matrix A_star
    """
    n = min(n_1, n_2)
    scaler = (1/np.sqrt(n)) if scaler is None else 1.0
    a_star, b_star = create_signal(n, s=r, distribution=distribution, Phi=None, scaler=scaler, seed=seed)
    b_star = np.abs(b_star)  # Sigular values need to be positives
    b_star = np.sort(b_star)[::-1]  # Sort in descending order
    Simga_star = np.zeros((n_1, n_2))
    Simga_star[:n, :n] = np.diag(b_star)  # Fill the diagonal with b*
    A_star = Simga_star
    if U is not None :
        A_star = U @ Simga_star
    if V is not None :
        A_star = A_star @ V.T

    #U, Sigma_star, V = np_SVD(A_star, type_svd='full') # (n_1, r), (r, r), (n_2, r)
    U, Sigma_star, VT = np.linalg.svd(A_star) # (n_1, n_1), (min(n_1, n_2),), (n_2, n_2)
    return A_star, U, Sigma_star, V

########################################################################################
########################################################################################

def get_matrices_UV(n_1, n_2, rank, symmetric=False, normalize=True, scale=None, seed=None, generator:torch.Generator=None):
    """
    A_star : (n_1, n_2)
    U_star : (n_1, n_1)
    Sigma_star : (min(n_1, n_2),)
    V_star : (n_2, n_2)
    """
    A_star, U_star, Sigma_star, V_star = get_matrices(n_1, n_2, rank, 1, symmetric, normalize, scale, seed, generator)
    return A_star.numpy(), U_star.numpy(), Sigma_star.numpy(), V_star.numpy()

def get_matrices_IndentityUV(n_1, n_2, r, distribution="normal", scaler=None, seed=None):
    """
    A_star : (n_1, n_2)
    U_star : (n_1, n_1)
    Sigma_star : (min(n_1, n_2),)
    V_star : (n_2, n_2)
    """
    # U = np.eye(n_1) # (n_1, n_1)
    # V = np.eye(n_2) # (n_2, n_2)
    U, V = None, None
    return create_matrix(n_1, n_2, r, U=U, V=V, scaler=scaler, seed=seed)

def get_matrices_QR_UV(n_1, n_2, r, distribution="normal", scaler=None, seed=None):
    """
    A_star : (n_1, n_2)
    U_star : (n_1, n_1)
    Sigma_star : (min(n_1, n_2),)
    V_star : (n_2, n_2)
    """
    U = create_orthonormal_basis(n_1, scaler=1.0) # (n_1, n_1), orthogonal
    V = create_orthonormal_basis(n_2, scaler=1.0) # (n_2, n_2), orthogonal
    # U = np.eye(n_1) # (n_1, n_1)
    # V = np.eye(n_2) # (n_2, n_2)
    return create_matrix(n_1, n_2, r, U=U, V=V, scaler=scaler, seed=seed)
    

########################################################################################
########################################################################################

def calculate_local_coherence(A, U_n1=None, V_n2=None):
    """
    Calculates the local coherence scores for a matrix A with respect to sets of vectors U_n1 and V_n2.

    Parameters:
    ----------
    A : np.ndarray of shape (n_1, n_2)
        The matrix for which to calculate local coherences.
    U_n1 : np.ndarray of shape (n_1, N_1), optional
        Set of N_1 vectors of dimension n_1 for left coherence calculation. If None, uses the canonical basis.
    V_n2 : np.ndarray of shape (n_2, N_2), optional
        Set of N_2 vectors of dimension n_2 for right coherence calculation. If None, uses the canonical basis.

    Returns:
    -------
    mu : np.ndarray of shape (N_1,)
        Local coherence values with respect to U_n1.
    nu : np.ndarray of shape (N_2,)
        Local coherence values with respect to V_n2.
    """
    n_1, n_2 = A.shape

    # Use canonical bases if U_n1 or V_n2 are not provided
    if U_n1 is None :
        U_n1 = np.eye(n_1) # (n_1, N_1) = (n_1, n_1)
    if V_n2 is None :
        V_n2 = np.eye(n_2) # (n_2, N_2) = (n_2, n_2)

    # SVD
    Sigma, U, V = np_SVD(A, type_svd='compact') # (r, r), (n_1, r), (n_2, r)

    r = Sigma.shape[0]

    mu = (n_1/r) * np.linalg.norm(U_n1.T @ U, axis=1)**2 # (N_1,)
    nu = (n_2/r) * np.linalg.norm(V_n2.T @ V, axis=1)**2 # (N_2,)

    P = mu[:,None] + nu[None,:] # P_{ij} = mu_i + nu_j
    #P = mu.view(-1, 1) + nu.view(1, -1) #  P_{ij} = mu_i + nu_j

    # Return local coherences value
    return mu, nu, P # (N_1,), (N_2,)

########################################################################################
########################################################################################

# TODO : control coherence

def generate_orthogonal_U_V(n_1, n_2, tau_1, tau_2):
    """
    Generate orthogonal matrices U (n_1 x n_1) and V (n_2 x n_2) with controlled coherence.

    Parameters:
        n1 (int): Number of rows/columns of U.
        n2 (int): Number of rows/columns of V.
        tau1 (float): Proportion of canonical basis-aligned singular vectors for U.
        tau2 (float): Proportion of canonical basis-aligned singular vectors for V.

    Returns:
        U (numpy array): Orthogonal matrix of shape (n1, n1).
        V (numpy array): Orthogonal matrix of shape (n2, n2).
    """
    # Number of canonical vectors
    N_1 = int(np.floor(tau_1 * n_1))
    N_2 = int(np.floor(tau_2 * n_2))

    variance_1 = 1/n_1
    variance_2 = 1/n_2

    # Construct U
    U = np.eye(n_1) * np.sqrt(variance_1)  # Start with the identity matrix
    if N_1 < n_1:  # If there are non-canonical components
        # Replace the remaining (n_1 - N_1) columns with random orthogonal vectors
        Q = np.linalg.qr(np.random.randn(n_1, n_1 - N_1) * np.sqrt(variance_1) )[0]
        U[:, N_1:] = Q[:, :n_1 - N_1]  # Replace columns after the first N_1

    # Ensure U is orthogonal
    U, _ = np.linalg.qr(U)

    # Construct V
    V = np.eye(n_2) * np.sqrt(variance_2)  # Start with the identity matrix
    if N_2 < n_2:  # If there are non-canonical components
        # Replace the remaining (n_2 - N_2) columns with random orthogonal vectors
        Q = np.linalg.qr(np.random.randn(n_2, n_2 - N_2) * np.sqrt(variance_2) )[0]
        V[:, N_2:] = Q[:, :n_2 - N_2]  # Replace columns after the first N_2

    # Ensure V is orthogonal
    V, _ = np.linalg.qr(V)

    return U, V



def F_numpy(A, X1X2=None, X2_bullet_X1=None) :
    """
    A : (n_1, n_2)
    X1 : (N, n_1)
    X2 : (N, n_2)
    X2_bullet_X1 : (N, n_1 x n_2)
    """
    if X2_bullet_X1 is not None :
        y_star = X2_bullet_X1 @ np_vec(A) # (N, n_1 x n_2) x (n_1 x n_2,) = (N,)
    else :
        X1, X2 = X1X2
        #y_star =  np.array([ np.inner(X1[s], A @ X2[s]) for s in range(X1.shape[0])]) # (N,)
        y_star =  np.array([X1[s].T @ A @ X2[s] for s in range(X1.shape[0])]) # (N,)
        # X2_bullet_X1 = face_splitting_product_numpy(X2, X1) # (N, n_1 x n_2)
        # y_star = X2_bullet_X1 @ np_vec(A) # (N, n_1 x n_2) x (n_1 x n_2,) = (N,)
    return y_star # (N,)
'''''
def F_cvxpy(A, X1X2=None, X2_bullet_X1=None):
    """
    Compute the linear mapping applied to A:
    A : (n_1, n_2)
    X1 : (N, n_1)
    X2 : (N, n_2)
    X2_bullet_X1 : (N, n_1 x n_2)
    """
    if X2_bullet_X1 is not None:
        # Use CVXPY symbolic operations here
        return X2_bullet_X1 @ cp.vec(A)  # Ensure A is treated as a CVXPY variable
    else:
        # Symbolic computation for individual terms
        X1, X2 = X1X2
        return cp.vstack([cp.sum(cp.multiply(X1[s][:, None] @ X2[s][None, :], A)) for s in range(X1.shape[0])])
'''



import numpy as np
import torch
import cvxpy as cp

def F_cvxpy(A, X1=None, X2=None, X1X2=None, X2_bullet_X1=None):
    """
    Compute <X1_i @ X2_i^T, A> for each i.

    A : np.ndarray, torch.Tensor, or cp.Variable / cp.Expression
    X1, X2 : arrays or tensors with shape (batch, n1) and (batch, n2) OR
             for convenience accept X1 entries as (batch, n1) and X2 as (batch, n2)
             and we treat outer product x1[:,None] @ x2[None,:].
    X1X2 : tuple (X1, X2)
    X2_bullet_X1 : optional precomputed vectorized mapping (batch, n1*n2)

    Returns:
        - If A is a cvxpy Variable/Expression: returns a column cvxpy expression of shape (batch, 1)
        - If A is torch.Tensor: returns torch.tensor shape (batch,)
        - If A is np.ndarray: returns numpy array shape (batch,)
    """
    # If precomputed vectorized mapping is given (X2_bullet_X1 @ vec(A))
    if X2_bullet_X1 is not None:
        # CVXPY case
        if isinstance(A, (cp.Variable, cp.Expression)):
            # X2_bullet_X1 should be either np.ndarray or cp.Constant; ensure it's compatible
            return (X2_bullet_X1 @ cp.vec(A)).reshape((-1, 1))
        else:
            # numeric: flatten A and multiply
            vecA = A.flatten()
            return X2_bullet_X1 @ vecA

    # unpack X1, X2 if provided as tuple
    if X1 is None or X2 is None:
        X1, X2 = X1X2

    # --- CVXPY symbolic case ---
    if isinstance(A, (cp.Variable, cp.Expression)):
        exprs = []
        # ensure X1, X2 are numpy arrays for safe use in cvxpy expressions
        for i in range(int(np.asarray(X1).shape[0])):
            x1_i = np.asarray(X1[i]).reshape(-1)   # (n1,)
            x2_i = np.asarray(X2[i]).reshape(-1)   # (n2,)
            # x1_i[:,None] @ x2_i[None,:] is a numeric (n1,n2) matrix
            # multiply elementwise with A (cvxpy var) then sum -> scalar
            expr_i = cp.sum(cp.multiply(x1_i[:, None] @ x2_i[None, :], A))
            exprs.append(expr_i)
        # return column vector (batch,1) => compatible with y shaped (batch,1)
        return cp.vstack(exprs)

    # --- PyTorch numeric case ---
    if isinstance(A, torch.Tensor):
        # expect X1, X2 numeric either torch or numpy; convert to torch if needed
        X1_t = X1 if isinstance(X1, torch.Tensor) else torch.tensor(np.asarray(X1), dtype=A.dtype, device=A.device)
        X2_t = X2 if isinstance(X2, torch.Tensor) else torch.tensor(np.asarray(X2), dtype=A.dtype, device=A.device)
        batch = X1_t.shape[0]
        out = torch.zeros(batch, dtype=A.dtype, device=A.device)
        # compute sum over elementwise (x1 outer x2) * A  -> scalar per batch
        for i in range(batch):
            # outer = x1_i[:,None] @ x2_i[None,:]
            outer = X1_t[i].unsqueeze(1) @ X2_t[i].unsqueeze(0)  # (n1,n2)
            out[i] = torch.sum(outer * A)
        return out

    # --- NumPy numeric case ---
    if isinstance(A, np.ndarray):
        X1_a = np.asarray(X1)
        X2_a = np.asarray(X2)
        batch = X1_a.shape[0]
        out = np.zeros(batch, dtype=A.dtype)
        for i in range(batch):
            outer = np.outer(X1_a[i], X2_a[i])  # (n1, n2)
            out[i] = np.sum(outer * A)
        return out

    raise ValueError(f"Unsupported type for A: {type(A)}")

########################################################################################
########################################################################################

# Measures

def get_measures_matrix_completion(A_star, N, X1=None, X2=None, P=None, tau=0, one_hot=False, shuffle=True, seed=None):
    """
    Take a matrix A_star and generate two sets of measures (X1, X2, y_star) and (X1_bar, X2_bar, y_star_bar) 
    such that the first set contains N elements and the second set contains the remaining elements.

    y[i] = X1[i]^T A_star X2[i] if one_hot=True else A_star[X1[i], X2[i]]

    For X1, X2, y_star :
        * If P is given, the first N_tau = tau * N elements are selected in the order of decreasing values in P.
        * If tau < 1 (i.e., N_tau < N), the remaining N - N_tau elements are selected randomly.
    For X1_bar, X2_bar, y_star_bar :
        * They contain the remaining elements not selected in X1, X2, y_star.
    """
    assert 0 <= tau <= 1, "tau must be in [0, 1]"
    n_1, n_2 = A_star.shape
    n = n_1 * n_2
    assert 0 < N <= n, "N must be in (0, n_1 * n_2]"
    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)

    # X1X2, y_star = data_matrix_completion(torch.from_numpy(A_star), one_hot=one_hot) # (n, n_1), (n, n_2), (n,)
    # X1, X2 = X1X2[:,0], X1X2[:,1] # (n, n_1), (n, n_2)
    (X1, X2), y_star = data_tensor_completion_general(torch.from_numpy(A_star), one_hot=one_hot) # (n, n_1), (n, n_2), (n_1,)
    X1, X2, y_star = X1.numpy(), X2.numpy(), y_star.numpy() # (n, n_1), (n, n_2), (n,)

    
    if P is not None:
        # Select the N_tau elements with the largest values in P (and the remaining elements randomly if tau < 1)

        # Compute the indices of the elements with the largest values in P
        flat_indices = np.argsort(P, axis=None)[::-1]  # Indices of the elements in decreasing order of P
        N_tau = int(tau * N)  # Number of elements to select
        top_indices = flat_indices[:N_tau]  # Indices of the top N_tau elements

        # Select the remaining elements randomly if tau < 1 (i.e., N_tau < N)
        remaining_indices = flat_indices[N_tau:]
        if len(remaining_indices) > 0 and (N - N_tau) > 0:
            selected_indices = np.concatenate([top_indices, np.random.choice(remaining_indices, N - N_tau, replace=False)])
        else :
            selected_indices = top_indices

    else:
        # If P is not given, select the first N elements
        if shuffle:
            selected_indices = np.random.permutation(y_star.shape[0])[:N]
        else:
            selected_indices = np.arange(N)

    # Select the remaining elements
    test_indices = np.setdiff1d(np.arange(y_star.shape[0]), selected_indices)
    X1_bar, X2_bar, y_star_bar = X1[test_indices], X2[test_indices], y_star[test_indices]  # (n_1*n_2-N, n_1), (n_1*n_2-N, n_2), (n_1*n_2-N,)

    # Select the N elements
    X1, X2, y_star = X1[selected_indices], X2[selected_indices], y_star[selected_indices]  # (N, n_1), (N, n_2), (N,)

    if seed is not None:
        np.random.set_state(old_state)

    return (X1, X2, y_star), (X1_bar, X2_bar, y_star_bar)

def get_measures_matrix_sensing(A_star, U_star, V_star, N, tau=0.0, variance=None, seed=None):
    M1, X1 = get_measures(N, Phi=U_star, tau=tau, variance=variance, seed=seed)
    M2, X2 = get_measures(N, Phi=V_star, tau=tau, variance=variance, seed=seed)
    X2_bullet_X1 = None

    X2_bullet_X1 = face_splitting_product_numpy(X2, X1)
    #y_star = F_cvxpy(A_star, X1X2=(X1, X2), X2_bullet_X1=X2_bullet_X1)
    y_star = F_numpy(A_star, X1X2=(X1, X2), X2_bullet_X1=X2_bullet_X1)

    return X1, X2, X2_bullet_X1, y_star

def get_data_matrix_factorization(A_star, U_star, V_star, N, problem, tau=0.0, variance=None, seed=None) :
    assert problem in ['matrix-completion', 'matrix-sensing']
    if problem == 'matrix-sensing':
        X1, X2, X2_bullet_X1, y_star = get_measures_matrix_sensing(A_star, U_star, V_star, N, tau, variance, seed=seed)
        X1_bar, X2_bar, X2_bullet_X1_bar, y_star_bar = None, None, None, None
    elif problem == 'matrix-completion':
        P = None
        if tau != 0 :
            mu, nu, P = calculate_local_coherence(A_star, U_n1=None, V_n2=None)
        (X1, X2, y_star), (X1_bar, X2_bar, y_star_bar) = get_measures_matrix_completion(A_star, N, P=P, tau=tau, one_hot=True, shuffle=True, seed=seed)

        X2_bullet_X1 = None
        X2_bullet_X1 = face_splitting_product_numpy(X2, X1)
        #y_star = F_cvxpy(A_star, X1X2=(X1, X2), X2_bullet_X1=X2_bullet_X1)
        y_star = F_numpy(A_star, X1X2=(X1, X2), X2_bullet_X1=X2_bullet_X1)

        if X1_bar.shape[0] > 0 :
            X2_bullet_X1_bar = face_splitting_product_numpy(X2_bar, X1_bar)
            #y_star_bar = F_cvxpy(A_star, X1X2=(X1_bar, X2_bar), X2_bullet_X1=X2_bullet_X1_bar)
            y_star_bar = F_numpy(A_star, X1X2=(X1_bar, X2_bar), X2_bullet_X1=X2_bullet_X1_bar)
        else :
            X2_bullet_X1_bar, y_star_bar = None, None

    return (X1, X2, X2_bullet_X1, y_star), (X1_bar, X2_bar, X2_bullet_X1_bar, y_star_bar)

########################################################################################
########################################################################################

# Convex optimization

def solve_matrix_factorization_nuclear_norm(n_1, n_2, y_star, X1X2=None, X2_bullet_X1=None, X1X2_bar=None, X2_bullet_X1_bar=None, reg=0, EPSILON=1e-6):
    """
    Solve the minimization problem (P5) to recover A*
    # Minimize ||A||_* subject to ||(X2 • X1)A - y*||_2 = sum_i (X1_i^T A X2_i - y*_i)^2 <= epsilon (epsilon = 0 in noiseless case
    
    For matrix completion, we want to solve
        Minimize ||A||_* subject to ||(X2 • X1)A - y*||_2 <= epsilon, 
        ie A_{ij} = A_{ij}^* for (i, j) in Omega
        or, if reg != 0, then
        Minimize ||A||_* subject to ||(X2 • X1)A - y*||_2 + reg * ||(X2_bar • X1_bar)A||_2 <= epsilon, ie A_{ij} = A_{ij}^* for (i, j) in Omega
        ie A_{ij} = A_{ij}^* for (i, j) in Omega and A_{ij} = 0 for (i, j) not in Omega
    """
    # Define the variable A (n_1 x n_2 matrix to be optimized)
    A = cp.Variable((n_1, n_2))

    # Define the objective function (nuclear norm of A)
    objective = cp.Minimize(cp.normNuc(A))

    # Define the constraints 
    if X1X2_bar is None and X2_bullet_X1_bar is None :
        reg = 0
    else :
        # If X1X2_bar and X2_bullet_X1_bar are not None, and reg is not 0, then the constraint is added
        # N_prime = X1X2_bar[0].shape[0] if X1X2_bar is not None else X2_bullet_X1_bar.shape[0]
        N_prime = 0
        N_prime = 0 if X1X2_bar[0] is None else X1X2_bar[0].shape[0]
        N_prime = 0 if X2_bullet_X1_bar is None else X2_bullet_X1_bar.shape[0]
        reg = reg if N_prime == 0 else 0
    #constraints = [F_cvxpy(A, X1X2=X1X2, X2_bullet_X1=X2_bullet_X1) = y_star]
    constraints = [
        cp.norm(F_cvxpy(A, X1X2=X1X2, X2_bullet_X1=X2_bullet_X1) - y_star, 'fro')
        + (reg * cp.norm(F_cvxpy(A, X1X2=X1X2_bar, X2_bullet_X1=X2_bullet_X1_bar), 'fro') if reg!=0 else 0) <= EPSILON]  # tolerance for numerical precision

    # Set up and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Solution
    A = A.value
    return A