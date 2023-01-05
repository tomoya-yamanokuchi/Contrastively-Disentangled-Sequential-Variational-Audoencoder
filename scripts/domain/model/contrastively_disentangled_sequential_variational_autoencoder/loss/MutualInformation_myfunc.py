import math
from matplotlib.pyplot import axis
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
compute
       log q(z)
    ~= log 1/(NM) sum_m=1^M q(z|x_m)
     = - log(MN) + logsumexp_m(q(z|x_m))
'''

class MutualInformation_myfunc(nn.Module):
    def __init__(self, num_train: int):
        super(MutualInformation_myfunc, self).__init__()
        self.num_train = num_train


    def logsumexp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0),
                                        dim=dim, keepdim=keepdim))
        else:
            raise ValueError('Must specify the dimension.')


    def log_density_f(self, mean, logvar, sample):
        num_batch, dim = mean.shape
        # << reshape >>
        mean   =   mean.view(num_batch, 1, dim)
        sample = sample.view(1, num_batch, dim)
        logvar = logvar.view(1, num_batch, dim)
        # << calculate logdensity >>
        y                            = (sample - mean)
        squared_mahalanobis_distance = torch.sum(((y**2) * torch.exp(-logvar)), axis=-1)
        log_det                      = torch.sum(logvar, axis=-1)
        const                        = torch.Tensor([dim * np.log(2 * np.pi)]).type_as(sample.data)
        return -0.5 * (const + log_det + squared_mahalanobis_distance)


    def log_density_z(self, mean, logvar, sample):
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


    def forward(self, f_dist, z_dist):
        assert type(f_dist) == tuple
        assert type(z_dist) == tuple
        '''
        f_mean, f_logvar, f_sample = f_dist
        z_mean, z_logvar, z_sample = z_dist
        '''

        # constant term: log(N*M)
        num_batch     = z_dist[0].shape[0]
        logNM         = torch.Tensor([np.log(self.num_train * num_batch)]).type_as(f_dist[0]) # ~= 13.9570 when N=9000, M=128

        # content entropy: H(f)
        log_qf_matrix = self.log_density_f(*f_dist)
        logsumexp_qf  = self.logsumexp(log_qf_matrix, dim=-1, keepdim=True) # sum over inner minibach (index j)
        Hf            = - torch.mean(logsumexp_qf.squeeze() - logNM)

        # motion entropy: H(z)
        log_qz_matrix = self.log_density_z(*z_dist)
        logsumexp_qz  = self.logsumexp(log_qz_matrix, dim=-1, keepdim=True) # sum over inner minibach (index j)
        Hz            = - torch.mean(logsumexp_qz.squeeze() - logNM)

        # joint entropy of content and motion: H(fz)
        logsumexp_qfz = self.logsumexp(log_qf_matrix + log_qz_matrix, dim=-1, keepdim=True)
        Hfz           = - torch.mean(logsumexp_qfz.squeeze() - logNM)

        # mutual information: I(f;z)
        mutual_info   = F.relu(Hf + Hz - Hfz)
        return mutual_info
