"""
This module provides functions to compute various norms of a model or its parameters.
"""

import torch 
import torch.nn as nn
import numpy as np


def l0_norm(W, threshold=1e-4, proportion=False):
    """Computes the l0 norm of a matrix W, defined as the number of non-zero elements.
    W : np.ndarray
        Matrix for which to compute the l0 norm.
    threshold : float
        Threshold value to consider an element as non-zero.
    proportion : bool
        If True, return the proportion of non-zero elements instead of the count.
    """
    #s = len(np.where(W.flatten() > threshold)[0])
    s = (np.abs(W) > threshold).sum().item()
    return s/np.prod(W.shape) if proportion else s

def l0_norm_model(model, threshold=1e-4, proportion=False, only_weights=True, requires_grad=False):
    """Computes the l0 norm of a model, defined as the number of non-zero elements.
    model : nn.Module
        Model for which to compute the l0 norm.
    threshold : float
        Threshold value to consider an element as non-zero.
    proportion : bool
        If True, return the proportion of non-zero elements instead of the count.
    only_weights : bool
        If True, only consider only the weights of the model.
    requires_grad : bool
        If True, only consider the parameters that require gradients.
    """
    if only_weights :
        if requires_grad :
            W = torch.cat([p.data.detach().cpu().flatten() for name, p in model.named_parameters() if 'weight' in name and p.requires_grad])
        else :
            W = torch.cat([p.data.detach().cpu().flatten() for name, p in model.named_parameters() if 'weight' in name])
    else :
        if requires_grad :
            W = torch.cat([p.data.detach().cpu().flatten() for p in model.parameters() if p.requires_grad])
        else :
            W = torch.cat([p.data.detach().cpu().flatten() for p in model.parameters() ])

    return l0_norm(W.numpy(), threshold=threshold, proportion=proportion)


def nuclear_norm_model(model, only_weights=True, requires_grad=False):
    """Computes the nuclear norm of a model, defined as the sum of the singular values.
    model : nn.Module
        Model for which to compute the nuclear norm.
    only_weights : bool
        If True, only consider only the weights of the model.
    requires_grad : bool
        If True, only consider the parameters that require gradients.
    """
    if only_weights :
        if requires_grad :
            return sum([torch.linalg.norm(p.data.detach().cpu(), ord='nuc') for name, p in model.named_parameters() if 'weight' in name and p.requires_grad])
        else :
            return sum([torch.linalg.norm(p.data.detach().cpu(), ord='nuc') for name, p in model.named_parameters() if 'weight' in name])
    else :
        if requires_grad :
            return sum([torch.linalg.norm(p.data.detach().cpu(), ord='nuc') for p in model.parameters() if p.requires_grad])
        else :
            return sum([torch.linalg.norm(p.data.detach().cpu(), ord='nuc') for p in model.parameters() ])

def l_p_norm_model(model, p=2, only_weights=True, requires_grad=False, concat_first=True):
    """Computes the l_p norm of a model, defined as the sum of the p-norms of the weights.
    model : nn.Module
        Model for which to compute the l_p norm.
    p : int
        Order of the norm.
    only_weights : bool
        If True, only consider only the weights of the model.
    requires_grad : bool
        If True, only consider the parameters that require gradients.
    concat_first : bool
        If True, concatenate the weights before computing the norm.
    """
    if concat_first :
        if only_weights :
            if requires_grad :
                W = torch.cat([p.data.detach().cpu().flatten() for name, p in model.named_parameters() if 'weight' in name and p.requires_grad])
            else :
                W = torch.cat([p.data.detach().cpu().flatten() for name, p in model.named_parameters() if 'weight' in name])
        else :
            if requires_grad :
                W = torch.cat([p.data.detach().cpu().flatten() for p in model.parameters() if p.requires_grad])
            else :
                W = torch.cat([p.data.detach().cpu().flatten() for p in model.parameters() ])
        return torch.norm(W.flatten(), p=p)
    else :
        if only_weights :
            if requires_grad :
                return sum([torch.linalg.norm(p.data.detach().cpu().flatten(), ord=p) for name, p in model.named_parameters() if 'weight' in name and p.requires_grad])
            else :
                return sum([torch.linalg.norm(p.data.detach().cpu().flatten(), ord=p) for name, p in model.named_parameters() if 'weight' in name])
        else :
            if requires_grad :
                return sum([torch.linalg.norm(p.data.detach().cpu().flatten(), ord=p) for p in model.parameters() if p.requires_grad])
            else :
                return sum([torch.linalg.norm(p.data.detach().cpu().flatten(), ord=p) for p in model.parameters() ])
            


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(10, 20, bias=False),
        nn.ReLU(),
        nn.Linear(20, 40, bias=False),
    )
    l0_norm = l0_norm_model(model, threshold=1e-4, proportion=False, only_weights=True, requires_grad=False)
    l1_norm = l_p_norm_model(model, p=1, only_weights=True, requires_grad=False, concat_first=True)
    l2_norm = l_p_norm_model(model, p=2, only_weights=True, requires_grad=False, concat_first=True)
    nuclear_norm = nuclear_norm_model(model, only_weights=True, requires_grad=False)
    
    print(f"L0 norm: {l0_norm}")
    print(f"L1 norm: {l1_norm}")
    print(f"L2 norm: {l2_norm}")
    print(f"Nuclear norm: {nuclear_norm}")