import torch
from torch import Tensor


def reparameterize(mean: Tensor, logvar: Tensor, random_sampling: bool=True) -> Tensor:
    if not random_sampling:
        return mean
    else:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + (eps * std)