import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import scipy
import cvxpy as cp
from tqdm import tqdm

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# print(default_colors)

########################################################################################
########################################################################################

"""
Basis $\Phi$ for compressed sensing
We want a matrix $\Phi \in \mathbb{R}^{n \times m}$ with $\Phi \overset{iid}{\sim} \mathcal{N}(0, \sigma^2)$ such that $\Phi^\top \Phi \propto \mathbb{I}_n$.

If $\Phi \overset{iid}{\sim} \mathcal{N}(0, \sigma^2)$, then $\mathbb{E}[\Phi^\top \Phi] = n \sigma^2 \mathbb{I}_n$ since
$\mathbb{E}[\Phi^\top \Phi]_{i,j}
= \sum_{k=1}^n  \mathbb{E}[\Phi_{ki} \Phi_{kj}]
= \sum_{k=1}^n \delta_{ij} \mathbb{E}[\Phi_{ki}^2]
= \delta_{ij} n \sigma^2$.

So $$\mathbb{E}[\Phi^\top \Phi] = \mathbb{I}_n \text{ for } \sigma = 1/\sqrt{n}$$

* Fourier basis
* $Q$ of the QR decomposition on random normal : Haar mesures
* Sample $\Psi \overset{iid}{\sim} \mathcal{N}(0, \sigma^2)$ and take $\Phi = \Psi \left( \Psi^\top \Psi\right)^{-\frac{1}{2}}$ : Haar mesures
"""

def create_Fourier_basis(n):
    """
    Creates a complex-valued matrix Φ of size (n, n) where each element is defined by:
        Φ_{ji} = (1 / sqrt(n)) * exp(-2 * π * i * j * i / n)

    Parameters:
    ----------
    n : int
        The dimension of the square matrix Φ.

    Returns:
    -------
    np.ndarray
        A complex-valued (n, n) matrix Φ with each element calculated as
        (1 / sqrt(n)) * exp(-2 * π * i * j * i / n).
    """
    #i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    #Phi = np.exp(-2j * np.pi * i * j / n)
    Phi = np.fft.fft(np.eye(n))
    return Phi / np.sqrt(n)  # normalized

def create_orthonormal_basis(n, scaler=None, seed=None):
    """
    Creates an orthonormal basis from a QR decomposition of a random matrix

    Parameters:
    ----------
    n : int
        The dimension of the matrix.

    Returns:
    -------
    np.ndarray
        A real-valued (n, n) matrix Φ, orthonormal.
    """
    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)
    Phi = np.random.randn(n, n)
    if seed is not None:
        np.random.set_state(old_state)
    scaler = (1/np.sqrt(n)) if scaler is None else 1.0
    Phi = scaler * Phi
    return np.linalg.qr(Phi)[0]  # orthonormalized
    
def create_normal_basis(n, scaler=None, seed=None, normalized=False):
    """
    Creates an orthonormal basis from the normalization of a random normal matrix

    Parameters:
    ----------
    n : int
        The dimension of the matrix.

    Returns:
    -------
    np.ndarray
        A real-valued (n, n) matrix Φ, orthonormal.
    """
    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)

    scaler = (1/np.sqrt(n)) if scaler is None else 1.0
    Phi = scaler * np.random.randn(n, n)

    if seed is not None:
        np.random.set_state(old_state)

    if normalized :
        Phi = Phi @ scipy.linalg.fractional_matrix_power(Phi.T @ Phi, -0.5) # ortho-normalized

    return Phi


########################################################################################
########################################################################################
### Signal $a^*$ and $b^*=\Phi a^*$

def create_signal(n, s, distribution="normal", Phi=None, scaler=None, seed=0):
    """
    Generate a sparse representation a*, with ||a*||_0 <= s

    Parameters:
    ----------
    n : int
        The dimension of the signal.
    s : int
        The sparsity level of the signal.
    distribution : str
        The distribution of the non-zero entries in a*.
    Phi : np.ndarray
        The basis matrix of shape (n, n).
    scaler : float
        The scaling factor for the signal, ie the standard deviation of the non-zero entries (default: 1/sqrt(n)).
    seed : int
        The random seed for reproducibility.

    Returns:
    -------
    np.ndarray
        A sparse representation a* of shape (n,).
    np.ndarray
        The corresponding signal b* = Φ a* of shape (n,).
    """
    assert distribution in ["normal", "uniform"]
    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)

    # a*
    a_star = np.zeros(n) # (n,)
    non_zero_indices = np.random.choice(n, s, replace=False) # (s,)
    a_star[non_zero_indices] = np.random.randn(s) if distribution=="normal" else np.random.choice(a=[-1, 0, 1], size=s, replace=True)
    # Make sure a*!=0
    while (a_star**2).sum() == 0 :
        a_star[non_zero_indices] = np.random.randn(s) if distribution=="normal" else np.random.choice(a=[-1, 0, 1], size=s, replace=True)

    if seed is not None:
        np.random.set_state(old_state)

    scaler = (1/np.sqrt(n)) if scaler is None else 1.0
    a_star = scaler * a_star
    b_star = a_star if Phi is None else Phi @ a_star
    return a_star, b_star


########################################################################################
########################################################################################
### Noise


def create_noise_from_scratch(N, SNR, n, s, distribution="normal", Phi=None, scaler=None, seed=0):
    """
    Create a noise vector ξ with the specified SNR = E||a*||_2^2 / E||ξ||_2^2

    Parameters:
    ----------
    N : int
        Dimension of the noise vector.
    n : int
        The dimension of the signal.
    s : int
        The sparsity level of the signal.
    SNR : float
        The signal-to-noise ratio. SNR = E||a*||_2^2 / E||ξ||_2^2
        SNR = np.inf : no noise
        SNR = 0 : only noise
    distribution : str
        The distribution of the non-zero entries in a*.
    Phi : np.ndarray
        The basis matrix of shape (n, n).
    scaler : float
        The scaling factor for the signal, ie the standard deviation of the non-zero entries (default: 1/sqrt(n)).
    seed : int
        The random seed for reproducibility.

    Returns:
    -------
    np.ndarray
        A noise vector ξ of shape (N,).
    """

    if SNR is None or SNR==np.inf :
        return np.zeros(N,)
    elif SNR==0 :
        return np.random.randn(N,) * np.inf

    maean_norm_a_star_square = np.mean([
        np.linalg.norm(create_signal(n, s, distribution, Phi=Phi, seed=None if seed is None else i*seed)[0])**2 
        for i in range(n * 10**2)
    ])
    sigma_xi = np.sqrt(maean_norm_a_star_square / (N * SNR))

    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)
    xi = sigma_xi * np.random.randn(N,) 
    if seed is not None:
        np.random.set_state(old_state)

    return xi

def create_noise_from_mean_norm_signal(N, SNR, maean_norm_a_star_square=None, seed=0):
    """
    Create a noise vector ξ with the specified SNR = E||a*||_2^2 / E||ξ||_2^2

    Parameters:
    ----------
    N : int
        Dimension of the noise vector.
    SNR : float
        The signal-to-noise ratio. SNR = E||a*||_2^2 / E||ξ||_2^2
        SNR = np.inf : no noise
        SNR = 0 : only noise
    seed : int
        The random seed for reproducibility.

    Returns:
    -------
    np.ndarray
        A noise vector ξ of shape (N,).
    """

    if SNR is None or SNR==np.inf :
        return np.zeros(N,)
    elif SNR==0 :
        return np.random.randn(N,) * np.inf

    sigma_xi = np.sqrt(maean_norm_a_star_square / (N * SNR))

    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)
    xi = sigma_xi * np.random.randn(N,) 
    if seed is not None:
        np.random.set_state(old_state)

    return xi

########################################################################################
########################################################################################
### Measures $X$

def get_measures(N, Phi, tau=0, variance=None, seed=None):
    """
    Generates a matrix M where r rows come from random columns of Phi,
    and the remaining N - r rows are random.
    Parameters:
    ----------
    Phi : np.ndarray
        Basis matrix of shape (n, m).
    N : int
        Number of rows for the generated matrix X.
    tau : float
        Proportion of rows in X that should be selected from columns of Phi.

    Returns:
    -------
    M : np.ndarray
        Generated matrix of shape (N, n).
    """
    assert 0 <= tau <= 1
    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(seed)

    n, m = Phi.shape
    if variance is None : variance = 1/n
    is_complex = np.iscomplexobj(Phi)

    if tau == 0 :
        M = np.random.randn(N, n) # (N, n)
        if is_complex : M = M + 1j * np.random.randn(N, n) # (N, n)
    else :
        if is_complex : M = np.zeros((N, n), dtype=complex)
        else : M = np.zeros((N, n))
        # Randomly choose tau*N columns from Phi to place in X
        N_1 = int(tau*N)
        #selected_columns = np.tile(np.arange(m), N_1 // m + 1)[:N_1] # repeat [0, ..., m-1] until the length is N_1
        if N_1<=m :
            #selected_columns = np.arange(N_1)
            selected_columns = np.random.choice(m, N_1, replace=False)
        else :
            selected_columns = np.zeros((N_1,), dtype=int)
            selected_columns[:m] = np.arange(m) # the first m columns are the columns of Phi
            selected_columns[m:] = np.random.choice(m, N_1-m, replace=True) # the last at sample randomnly from Phy
        # Fill the first r rows with random columns of Phi
        M[:N_1, :] = Phi[:, selected_columns].T * (1/np.sqrt(variance)) # Transpose to match dimensions (N_1, n)
        # Fill the remaining rows with random entries
        if is_complex :
            #for i in range(N_1, N): M[i, :] = np.random.randn(n) + 1j * np.random.randn(n)
            M[N_1:, :] = np.random.randn(N-N_1, n) + 1j * np.random.randn(N-N_1, n) # (N-N_1, n)
        else :
            #for i in range(N_1, N): M[i, :] = np.random.randn(n)
            M[N_1:, :] = np.random.randn(N-N_1, n) # (N-N_1, n)

    if seed is not None:
        np.random.set_state(old_state)

    M = np.sqrt(variance) * M
    X = M @ Phi # (N, m)
    return M, X # (N, n)


########################################################################################
########################################################################################
### Coherence

def calculate_coherence(A, B):
    """
    Calculates the coherence between the columns of two matrices A and B.

    Parameters:
    ----------
    A : np.ndarray
        Matrix of shape (q, m) where columns are the vectors to compare.
    B : np.ndarray
        Matrix of shape (q, m) where columns are the vectors to compare.

    Returns:
    -------
    float
        Maximum coherence between the columns of A and B.
    """
    # Normalize columns of A and B
    A_normalized = A / np.linalg.norm(A, axis=0, keepdims=True)
    B_normalized = B / np.linalg.norm(B, axis=0, keepdims=True)

    # Compute coherence matrix (absolute inner products between columns)
    coherence_matrix = np.abs(A_normalized.T @ B_normalized)

    # Set self-coherence to zero if A and B have the same columns
    if A.shape == B.shape and np.allclose(A, B):
        np.fill_diagonal(coherence_matrix, 0)

    # Return the maximum coherence value
    return np.max(coherence_matrix)

########################################################################################
########################################################################################
## Convex Programming

def solve_compressed_sensing_l1(X, y_star, EPSILON=1e-8):
    """
    Solve the l1-minimization problem to recover a :
    Minimize ||a||_1 subject to ||Xa - y*||_2 <= epsilon
    """
    a = cp.Variable(X.shape[1])
    objective = cp.Minimize(cp.norm(a, 1))
    #constraints = [X @ a = y]
    constraints = [cp.norm(X @ a - y_star, 2) <= EPSILON]  # tolerance for numerical precision
    problem = cp.Problem(objective, constraints)
    problem.solve()
    # Recovered sparse representation a
    return a.value