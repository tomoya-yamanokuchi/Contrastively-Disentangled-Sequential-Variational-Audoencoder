import torch
import numpy as np


def log_density_z(mean, logvar, sample):
    num_batch, step, dim = mean.shape
    # << reshape >>
    mean   =   mean.permute((1, 0, 2)).unsqueeze(-2) # stepの次元を最初に出してブロードキャスト用の次元を追加
    sample = sample.permute((1, 0, 2)).unsqueeze(1)  # stepの次元を最初に出してブロードキャスト用の次元を追加
    logvar = logvar.permute((1, 0, 2)).unsqueeze(1)  # stepの次元を最初に出してブロードキャスト用の次元を追加
    # << calculate logdensity >>
    y                            = (sample - mean)
    squared_mahalanobis_distance = torch.sum(((y**2) * torch.exp(-logvar)), axis=-1)
    log_det                      = torch.sum(logvar, axis=-1)
    const                        = torch.Tensor([dim * np.log(2 * np.pi)]).type_as(sample.data)
    return (-0.5 * (const + log_det + squared_mahalanobis_distance)).sum(axis=0) # sum over timestep