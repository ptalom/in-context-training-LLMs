import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
import warnings
from sklearn import tree
import xgboost as xgb
import cvxpy as cp
import numpy as np


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "matrix_completion": [
            (LeastSquaresModel, {}),
            (NuclearNormMinimizationModel, {"m": 5, "n": 5, "epsilon": 1e-6}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleave xs and ys, where ys_b is already T(c) of shape (B, N, d)."""
        bsize, points, dim = xs_b.shape

        # ys_b should be (B, N, d)
        assert ys_b.shape == (bsize, points, dim), \
            f"ys_b must be (B, N, d), got {ys_b.shape}"

        # (B, N, 2, d)
        zs = torch.stack((xs_b, ys_b), dim=2)

        # (B, 2N, d)
        zs = zs.view(bsize, 2 * points, dim)
        #print("xs_b:", xs_b.shape)
        #print("ys_b:", ys_b.shape)

        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        
        #print("output shape:", output.shape)
        return prediction[:, :ys.shape[1], :ys.shape[2]]



class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu() 
        B, L, D = xs.shape

        if inds is None:
            inds = range(L)
        else:
            if max(inds) >= L or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros(B, D))  # first point = zeros
                continue

            # train set pour tous les batches: [B, i, D]
            train_xs, train_ys = xs[:, :i, :], ys[:, :i, :]

            # test_x: [B, 1, D]
            test_x = xs[:, i : i + 1, :]

            # calcul de la pseudo-inverse batch: [B, D, i] @ [B, i, D] -> [B, D, D]
            # on reshape pour que bmm fonctionne: [B, D, i] pour pinverse @ [B, i, D]
            train_xs_t = train_xs.transpose(1, 2)  # [B, D, i]

            ws = torch.bmm(torch.linalg.pinv(train_xs), train_ys)  # [B, D, D] @ [B, i, D] ?

            # plus simple: Least squares vectorisé via pinv: ws = pinv(train_xs[b]) @ train_ys[b]
            # donc on fait batch bmm avec pseudo-inverse
            ws = torch.bmm(torch.linalg.pinv(train_xs), train_ys)  # [B, D, D] ?

            # pred = test_x @ ws
            pred = torch.bmm(test_x, ws)  # [B, 1, D]

            preds.append(pred[:, 0, :])  # [B, D]

        return torch.stack(preds, dim=1)  # [B, len(inds), D]



def flatten_onehot_to_rowcol(x_flat: np.ndarray, m: int, n: int):
    """
    x_flat: shape (d,) or (1,d) one-hot vector where d = m*n
    returns: (x1, x2) as numpy arrays shape (1,m), (1,n)
    """
    # support batch single vector or 1D
    x_flat = x_flat.reshape(-1)
    assert x_flat.size == m * n
    idx = int(np.argmax(x_flat))  # index in [0, m*n-1]
    row = idx // n
    col = idx % n
    x1 = np.zeros((1, m), dtype=x_flat.dtype)
    x2 = np.zeros((1, n), dtype=x_flat.dtype)
    x1[0, row] = 1.0
    x2[0, col] = 1.0
    return x1, x2

def F_cvxpy(A, X1, X2, is_variable=False):
    """
    Build CVXPY expression (or numpy evaluation) of F(A, X1, X2):
      For each j: f_j = x1_j @ A @ x2_j.T
    - If A is a CVXPY Variable (is_variable=True), returns a cp.vstack expression shape (i,1).
    - If A is numpy/torch array, returns a numpy array shape (i,1).
    Parameters
    ----------
    A: cp.Variable or numpy array
    X1: numpy array shape (i, n1) or torch -> treated as numeric
    X2: numpy array shape (i, n2)
    """
    # convert to numpy if torch
    if isinstance(X1, torch.Tensor):
        X1 = X1.cpu().numpy()
    if isinstance(X2, torch.Tensor):
        X2 = X2.cpu().numpy()

    i = X1.shape[0]
    outputs = []
    if is_variable:
        # A is a CVXPY variable
        for j in range(i):
            # make constants (1, n1) and (n2, 1)
            x1_c = cp.Constant(X1[j : j + 1, :])  # shape (1, n1)
            x2_c = cp.Constant(X2[j : j + 1, :]).T  # shape (n2, 1)
            # compute 1x1 expression
            outputs.append(x1_c @ A @ x2_c)  # cp expression (1,1)
        # stack vertically -> shape (i,1)
        return cp.vstack(outputs)
    else:
        # A is numeric (np.ndarray)
        A_np = np.asarray(A)
        for j in range(i):
            val = float((X1[j : j + 1, :] @ A_np @ X2[j : j + 1, :].T).reshape(-1)[0])
            outputs.append([val])
        return np.array(outputs).reshape(-1, 1)


class NuclearNormMinimizationModel:
    """
    Nuclear norm minimization model using CVXPY.
    Input:
      xs: torch.Tensor (batch, n_points, n_dim_in)  -- n_dim_in can be either:
            - m*n (one-hot indices flattened), OR
            - n1 + n2 (concatenated x1,x2)
      ys: torch.Tensor (batch, n_points) -- scalar observations
    Output:
      torch.Tensor (batch, len(inds))
    """

    def __init__(self, m, n, epsilon=1e-6, solver="SCS", reg=0):
        self.m = int(m)
        self.n = int(n)
        self.epsilon = float(epsilon)
        self.solver = solver
        self.reg = reg
        self.name = f"nuclear_norm_minimization_epsilon={self.epsilon}"

    def _xs_to_x1x2(self, xs_row):
        """
        Convert a single xs vector (numpy or torch) to x1 (1,m) and x2 (1,n).
        Handles both flattened one-hot of size m*n and concatenated form (m+n).
        """
        if isinstance(xs_row, torch.Tensor):
            xs_row = xs_row.cpu().numpy()
        xs_row = xs_row.reshape(-1)

        if xs_row.size == self.m * self.n:
            
            x1, x2 = flatten_onehot_to_rowcol(xs_row, self.m, self.n)
            return x1.astype("float32"), x2.astype("float32")
        elif xs_row.size == self.m + self.n:
            # concatenated [x1, x2]
            x1 = xs_row[: self.m].reshape(1, self.m)
            x2 = xs_row[self.m :].reshape(1, self.n)
            return x1.astype("float32"), x2.astype("float32")
        else:
            raise ValueError(
                f"Unexpected xs vector length {xs_row.size}; expected {self.m*self.n} or {self.m+self.n}"
            )

    def __call__(self, xs, ys, inds=None):
        xs = xs.cpu()
        ys = ys.cpu()
        batch_size, n_points, n_dim_in = xs.shape
        d = self.m * self.n  
        
        if inds is None:
            inds = range(n_points)
        
        # Init ys_pred à la forme (B, n_points, d)
        ys_pred = torch.zeros(batch_size, n_points, d, dtype=torch.float32)
        
        for b in range(batch_size):
            # Construire training set
            X1_train_list, X2_train_list, y_train_list = [], [], []
            for j in range(n_points - 1):  # dernier point = à prédire
                x1_j, x2_j = self._xs_to_x1x2(xs[b, j, :])
                X1_train_list.append(x1_j)
                X2_train_list.append(x2_j)
                y_train_list.append([float(ys[b, j, 0].item())])
            
            if len(X1_train_list) == 0:
                continue
            
            X1_train = np.vstack(X1_train_list)
            X2_train = np.vstack(X2_train_list)
            y_train = np.array(y_train_list).reshape(-1, 1)
            
            # CVXPY variable
            A_var = cp.Variable((self.m, self.n))
            F_expr = F_cvxpy(A_var, X1_train, X2_train, is_variable=True)
            y_const = cp.Constant(y_train)
            prob = cp.Problem(cp.Minimize(cp.normNuc(A_var)), [cp.norm(F_expr - y_const, 2) <= self.epsilon])
            try:
                prob.solve(solver=self.solver, verbose=False)
                A_val = np.array(A_var.value, dtype="float32")
            except:
                A_val = None
            
            # prédire pour tous les points
            for i in range(n_points):
                x1_test, x2_test = self._xs_to_x1x2(xs[b, i, :])
                if A_val is None:
                    pred_val = 0.0
                else:
                    pred_val = float(F_cvxpy(A_val, X1=x1_test, X2=x2_test, is_variable=False).reshape(-1)[0])
                ys_pred[b, i, 0] = pred_val  
        
        return ys_pred

