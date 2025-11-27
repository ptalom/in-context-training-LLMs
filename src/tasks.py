import math

import torch
import torch.nn as nn

from samplers import power_law_matrix


def squared_error(ys_pred, ys):
    # comparer uniquement la première dimension
    return ((ys_pred[..., 0] - ys[..., 0])**2)

def mean_squared_error(ys_pred, ys):
    return ((ys_pred[..., 0] - ys[..., 0])**2).mean()

"""

def squared_error(ys_pred, ys):
    squared_error = (ys - ys_pred).square()
    #print("squared_error:", squared_error)
    return squared_error


def mean_squared_error(ys_pred, ys):
    mean_squared_error = (ys - ys_pred).square().mean()
    #print("mean_squared_error:", ys_pred)
    return mean_squared_error
"""

def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()



sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "matrix_completion": MatrixCompletion,
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,

    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error



class MatrixCompletion(Task):
    """
    Task Matrix Completion.
    Evaluer f(x) = vec(M)^T x pour x_i ∈ {e_1,...,e_d}
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None,
                 sampling="uniform", hard=False, rank=2,
                 m=5, n=5, alpha=0.0, beta=0.0, device="cpu",
                 embedding_dim=16):
        self.n_dims = n_dims
        self.batch_size = batch_size
        self.m = m
        self.n = n
        self.rank = rank
        self.d = m * n
        self.sampling = sampling
        self.hard = hard
        self.device = device
        self.embedding_dim = embedding_dim

        # Générer M si pool_dict non fourni
        if pool_dict is None:
            seed = seeds[0] if seeds else None
            self.M = power_law_matrix(
                n_1=m, n_2=n, rank=rank, batch_size=1,
                symmetric=False, normalize=True, scale=None,
                alpha=alpha, beta=beta, seed=seed, device=device
            ).squeeze(0)
        else:
            self.M = pool_dict["M"]

        self.w = self.M.flatten()

        # Paramètres d'embedding
        #self.W = nn.Parameter(torch.randn(self.d, self.embedding_dim))  # one-hot embeddings
        #self.scalar_proj = nn.Linear(1, self.embedding_dim, bias=False)  # projection f(x_i) -> embedding

    def evaluate(self, xs_b, target_n_dims=None, target_n_points=None):
        """
        Évalue f(x) = vec(M)^T x pour un batch xs_b.

        Args:
            xs_b: tensor [B, n_points, n_dims]
            target_n_dims: int, dimension finale attendue par le modèle
            target_n_points: int, nombre de points attendu par le modèle

        Returns:
            ys_b: tensor [B, target_n_points, target_n_dims]
        """
        B, n_points, n_dims = xs_b.shape

        # Assurer que w est compatible avec xs_b
        w = self.w.to(xs_b.device)
        if w.numel() < n_dims:  # Pad w avec des zéros si trop petit
            w = torch.cat([w, torch.zeros(n_dims - w.numel(), device=xs_b.device)])
        elif w.numel() > n_dims:  # Tronquer si trop grand
            w = w[:n_dims]

        # Calculer les valeurs scalaires pour chaque point
        scalar_vals = torch.einsum('bnd,d->bn', xs_b, w).unsqueeze(-1)  # [B, n_points, 1]

        # Déterminer la taille finale
        target_n_points = target_n_points or n_points
        target_n_dims = target_n_dims or n_dims

        # Initialiser ys_b avec des zéros
        ys_b = torch.zeros((B, target_n_points, target_n_dims), device=xs_b.device)

        # Copier les valeurs scalaires dans la première dimension
        ys_b[:, :n_points, 0] = scalar_vals[:, :, 0]
        print("ys_b shape:", ys_b.shape)
        print("ys_1 : ", ys_b[0])

        return ys_b
    

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
"""
    def evaluate(self, xs_b, target_n_dims=None, target_n_points=None):
        ""
        Évalue f(x) = vec(M)^T x pour un batch xs_b
        et renvoie embeddings.

        Args:
            xs_b: tensor [B, n_points, n_dims]
            target_n_dims: int, dimension finale attendue par le modèle
            target_n_points: int, nombre de points attendu par le modèle
        Returns:
            ys_b: tensor [B, target_n_points, target_n_dims]
        ""
        B, n_points, n_dims = xs_b.shape
        device = xs_b.device

        # Assurer compatibilité w
        w = self.w.to(device)
        if w.numel() < n_dims:
            w = torch.cat([w, torch.zeros(n_dims - w.numel(), device=device)])
        elif w.numel() > n_dims:
            w = w[:n_dims]

        # Calculer valeurs scalaires
        scalar_vals = torch.einsum('bnd,d->bn', xs_b, w).unsqueeze(-1)  # [B, n_points, 1]

        # Embeddings one-hot
        E_x = torch.einsum('bnd,dD->bnD', xs_b, self.W)  # [B, n_points, embedding_dim]

        # Embeddings scalaires
        E_c = self.scalar_proj(scalar_vals)  # [B, n_points, embedding_dim]

        # Combinaison
        ys_b = E_x + E_c  # [B, n_points, embedding_dim]

        # Ajuster taille finale si besoin
        target_n_points = target_n_points or n_points
        target_n_dims = target_n_dims or self.embedding_dim

        if target_n_points > n_points:
            pad = torch.zeros(B, target_n_points - n_points, target_n_dims, device=device)
            ys_b = torch.cat([ys_b, pad], dim=1)
        if target_n_dims > self.embedding_dim:
            pad = torch.zeros(B, target_n_points, target_n_dims - self.embedding_dim, device=device)
            ys_b = torch.cat([ys_b, pad], dim=2)
        print("ys_b shape:", ys_b.shape)
        print("ys_1 : ", ys_b[0])
        return ys_b
"""


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error






