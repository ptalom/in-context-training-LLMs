import math
import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "matrix_completion": MatrixCompletionSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t

class MatrixCompletionSampler(DataSampler):
    """
    Sampler pour la matrix completion.
    Génère les vecteurs x_i ∈ {e_1,...,e_d} pour construire un prompt P.
    Supporte uniform, magnitude et coherence-based sampling.
    """
    def __init__(self, n_dims, m, n, rank=2, hard=False, sampling="uniform",
                 alpha=0.0, beta=0.0, seed=None, device="cpu"):
        self.n_dims = n_dims
        self.m = m
        self.n = n
        self.d = m * n
        self.rank = rank
        self.sampling = sampling
        self.hard = hard
        self.seed = seed
        self.device = device

        # Générer M avec power-law
        self.M = power_law_matrix(
            n_1=m, n_2=n, rank=rank, batch_size=1,
            symmetric=False, normalize=True, scale=None,
            alpha=alpha, beta=beta, seed=seed, device=device
        ).squeeze(0)

        self.w = self.M.flatten()

        # Probabilités de sampling
        if sampling == "uniform":
            self.probs = torch.ones(self.d, device=device) / self.d
        elif sampling == "coherence":
            mu, nu = local_coherences(self.M, r=rank)
            P_matrix = coherence_based_distribution(mu, nu, rank=rank, normalize=True)
            self.probs = P_matrix.flatten()
        elif sampling == "magnitude":
            self.probs = self.w.abs() / self.w.abs().sum()
        else:
            raise ValueError(f"Sampling inconnu {sampling}")

    def sample_xs(self, n_points, b_size, n_dims=None):
        """
        Génère les x_i pour construire le prompt P.
        x_i ∈ {e_1, ..., e_d}, batch_size = b_size
        """
        probs_b = self.probs.unsqueeze(0).repeat(b_size, 1)

        if self.hard:
            # Top indices
            top_indices = torch.topk(probs_b, n_points, dim=1).indices
        else:
            # Échantillonnage multinomial
            top_indices = torch.stack([
                torch.multinomial(probs_b[i], n_points, replacement=False)
                for i in range(b_size)
            ], dim=0)

        # One-hot encoding
        xs_b = torch.zeros(b_size, n_points, self.d, device=self.device)
        batch_idx = torch.arange(b_size).unsqueeze(1).repeat(1, n_points)
        point_idx = torch.arange(n_points).unsqueeze(0).repeat(b_size, 1)
        xs_b[batch_idx, point_idx, top_indices] = 1.0

        return xs_b
    
"""
class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
"""


def power_law_matrix(
    n_1:int, n_2:int, rank:int, batch_size:int=1,
    symmetric=False, normalize=True, scale=None,
    alpha:float=0.0, beta:float=0.0,
    seed=None, device: str = "cpu", dtype=torch.float32
) -> torch.Tensor:
    """
    Generate a batch of 'power-law' matrices of size (n_1, n_2) with a rank equal to 'rank':

        M = A U V^T B,  A_ii = i^{-alpha} and B_jj = j^{-beta}

    where U, V have i.i.d. N(0,1) entries.

    Parameters
    ----------
    n_1, n_2 : int
        First and second matrix dimensions
    rank : int
        Target rank.
    batch_size : int
        Number of matrices to generate
    symmetric : bool
        If True, the matrices are symmetric : if symmetric is True, n_1 must be equal to n_2
    normalize : bool
        If True, the matrices are normalized such that the Frobenius norm is equal to sqrt(d_1*d_2)
    scale : float
        Scale factor to apply to the matrices, if None, scale = 1/sqrt(rank)
        The singular values of the matrices are approximately equal to 'scale'
    alpha, beta : float
        Power-law exponents. alpha=beta=0 ~ incoherent, alpha=beta=1 ~ highly coherent.
    device : str
        'cpu' or 'cuda' device.
    dtype : torch.dtype
        Floating type for computations.
    normalize : bool
        Whether to normalize the matrix to have frobenius norm 1.

    Returns
    -------
    M : (n_1, n_2) torch.Tensor
        Generated low-rank matrix.
    """

    if seed :
        generator = torch.Generator()
        generator.manual_seed(seed)
    else :
        generator = None

    #assert r <= min(n_1, n_2), "r must be <= min(n_1, n_2)"
    assert not symmetric or n_1==n_2, "n_1 must be equal to n_2 if symmetric is True"

    U = torch.randn(batch_size, n_1, rank, device=device, dtype=dtype, generator=generator) # (batch_size, d_2, rank)

    if symmetric:
        V = U # (batch_size, d_1, rank)
    else:
        V = torch.randn(batch_size, n_2, rank, device=device, dtype=dtype, generator=generator) # (batch_size, d_2, rank)

    scaler = 1/math.sqrt(rank) if scale is None else scale
    M = torch.bmm(U, V.transpose(1, 2)) * scaler # (batch_size, n_1, rank) x (batch_size, rank, n_2) = (batch_size, n_1, n_2)

    if alpha!=0.0 :
        i = torch.arange(1, n_1 + 1, device=device, dtype=dtype) # (n_1,)
        A = torch.diag(i.pow(-alpha)) # (n_1, n_1), diag(i^{-alpha})
        M = torch.matmul(A, M) # (1, n_1, n_1) x (batch_size, n_1, n_2) = (batch_size, n_1, n_2)
    if beta!=0.0 :
        j = torch.arange(1, n_2 + 1, device=device, dtype=dtype)
        B = torch.diag(j.pow(-beta)) # (n_1, n_1), diag(j^{-beta})
        M = torch.matmul(M, B) # (batch_size, n_1, n_2) x (1, n_2, n_2)  = (batch_size, n_1, n_2)

    # Normalize for that the frobenius norm = 1
    if normalize:
        #M = M / M.norm()
        M = M / torch.norm(M, dim=(1, 2), p='fro', keepdim=True) # (batch_size, n_1, n_2)

    #return M.squeeze(0) if batch_size==1 else M
    return M


def local_coherences(M: torch.Tensor, r: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute local coherences (mu_i, nu_j) from the SVD of M, of shape (n_1, n_2)

        mu_i = (n_1/r) * ||U_i||_2^2
        nu_j = (n_2/r) * ||V_j||_2^2,

    where M = U diag(S) V^T.

    Parameters
    ----------
    M : (n_1, n_2) or (batch_size, n_1, n_2) torch.Tensor
        Input matrix.
    r : int or None
        Rank truncation. If None, r = min(n1, n2).

    Returns
    -------
    mu : (n_1,) or (batch_size, n_1) torch.Tensor
        Row local coherences.
    nu : (n_2,) or (batch_size, n_2) torch.Tensor
        Column local coherences.
    """
    original_ndim = M.dim()
    if original_ndim == 2:
        n_1, n_2 = M.shape
        M = M.unsqueeze(0) # (1, n_1, n_2)
        batch_size = 1
    else :
        batch_size, n_1, n_2 = M.shape

    if r is None:
        r = min(n_1, n_2)

    ## SVD
    U, Sigma, VT = torch.linalg.svd(M, full_matrices=False) # (batch_size, n_1, n), (batch_size, n), (batch_size, n, n_2) with n=min(n_1, n_2)
    V = VT.transpose(1, 2) # (batch_size, n, n_2)
    ## Compact SVD
    #Sigma = Sigma[:,:r] # (batch_size, r)
    U, V = U[:,:,:r], V[:,:,:r] # (batch_size, n_1, r), (batch_size, n_2, r)

    # squared row norms
    mu = (n_1 / float(r)) * (U ** 2).sum(dim=-1) # (batch_size, n_1)
    nu = (n_2 / float(r)) * (V ** 2).sum(dim=-1) # (batch_size, n_2)

    return (mu.squeeze(0), nu.squeeze(0)) if original_ndim == 2 else (mu, nu)


def coherence_based_distribution(mu: torch.Tensor, nu: torch.Tensor, rank, c0=1.0, upper_bound=False, lower_bound=False, normalize=False) -> torch.Tensor:
    """
    Build a probability matrix P_ij proportional to mu_i + nu_j:

        P_ij >= min{ c0 * (mu_i + nu_j) * r * log^2(n_1+n_2) / min(n_1, n_2), 1 }
        p_ij >= 1 / min(n_1, n_2)^10

    Parameters
    ----------
    mu : (n_1,) or (batch_size, n_1) torch.Tensor
        Row local coherences.
    nu : (n_2,) or (batch_size, n_2) torch.Tensor
        Column local coherences.
    rank : int
        Rank of the matrices
    c0 : float
        Normalization constant.
    upper_bound : bool
        Whether to clip P_ij >= 1.
    lower_bound : bool
        Whether to clip P_ij <= 1 / min(n_1, n_2)^10.
    normalize : bool
        Whether to normalize the matrix to have probability distribution

    Returns
    -------
    P : (n_1, n_2) or (batch_size, n_1, n_2) torch.Tensor
        Probability matrix
    """
    original_ndim = mu.dim()
    if original_ndim == 1:
        n_1, n_2 = mu.shape[0], nu.shape[0]
        batch_size = 1
        mu, nu = mu.unsqueeze(0), nu.unsqueeze(0) # (1, n_1), (1, n_2)
    else :
        batch_size, n_1 = mu.shape
        assert nu.shape[0] == batch_size, "mu and nu must have the same batch size"
        n_2 = nu.shape[1]

    # W_{ij} = mu_i + nu_j
    W = mu.view(batch_size, n_1, 1) + nu.view(batch_size, 1, n_2)  # (batch_size, n_1, n_2)

    n = min(n_1, n_2)
    P = c0 * rank * W * math.log(n_1 + n_2) ** 2 / float(n)
    if upper_bound : P = torch.clamp(P, 0.0, 1.0) # make sure P_ij <= 1.0
    if lower_bound : P = torch.clamp(P, 1.0 / (n ** 10), 1.0) # make sure P_ij >= 1 / n^10

    # normalize
    if normalize : P = P / P.sum(dim=(-2,-1), keepdim=True)


    return P.squeeze(0) if original_ndim == 1 else P

