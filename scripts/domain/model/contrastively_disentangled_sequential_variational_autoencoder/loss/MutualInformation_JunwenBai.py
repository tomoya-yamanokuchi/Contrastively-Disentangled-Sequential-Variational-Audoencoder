import math
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

class MutualInformation_JunwenBai(nn.Module):
    def __init__(self, num_train: int):
        super(MutualInformation_JunwenBai, self).__init__()
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


    def log_density(self, mean, logvar, sample):
        mu       = mean
        logsigma = logvar
        # -----------------------------------
        mu = mu.type_as(sample)
        logsigma = logsigma.type_as(sample)
        c = torch.Tensor([np.log(2 * np.pi)]).type_as(sample.data)

        inv_sigma = torch.exp(-logsigma)
        tmp = (sample - mu) * inv_sigma
        return -0.5 * (tmp * tmp + 2 * logsigma + c)


    # def log_density(self, mean, logvar, sample):
    #     c          = torch.Tensor([np.log(2 * np.pi)]).type_as(sample.data)
    #     return -0.5 * (c + logvar + ((sample - mean)**2 * torch.exp(-logvar)))


    def forward(self, f_dist, z_dist):
        assert type(f_dist) == tuple
        assert type(z_dist) == tuple
        '''
        f_mean, f_logvar, f_sample = f_dist
        z_mean, z_logvar, z_sample = z_dist
        '''

        num_batch, step, dim_z = z_dist[0].shape
        num_batch, dim_f       = f_dist[0].shape

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        # num_batch x num_batch x dim_f
        _logq_f_tmp = self.log_density(
            mean   = f_dist[0].unsqueeze(0).repeat(step, 1, 1).view(step, 1, num_batch, dim_f), # [8, 1, 128, 256]
            logvar = f_dist[1].unsqueeze(0).repeat(step, 1, 1).view(step, 1, num_batch, dim_f), # [8, 1, 128, 256]
            sample = f_dist[2].unsqueeze(0).repeat(step, 1, 1).view(step, num_batch, 1, dim_f), # [8, 128, 1, 256]
        )

        # step x num_batch x num_batch x dim_f
        _logq_z_tmp = self.log_density(
            mean   = z_dist[0].transpose(0, 1).view(step, 1, num_batch, dim_z), # [8, 1, 128, 32]
            logvar = z_dist[1].transpose(0, 1).view(step, 1, num_batch, dim_z), # [8, 1, 128, 32]
            sample = z_dist[2].transpose(0, 1).view(step, num_batch, 1, dim_z), # [8, 128, 1, 32]
        )

        import ipdb; ipdb.set_trace()
        _logq_fz_tmp = torch.cat((_logq_f_tmp, _logq_z_tmp), dim=3) # [8, 128, 128, 288]

        logq_f  = (self.logsumexp(_logq_f_tmp.sum(3), dim=2, keepdim=False)  - math.log(num_batch * self.num_train)) # [8, 128]
        logq_z  = (self.logsumexp(_logq_z_tmp.sum(3), dim=2, keepdim=False)  - math.log(num_batch * self.num_train)) # [8, 128]
        logq_fz = (self.logsumexp(_logq_fz_tmp.sum(3), dim=2, keepdim=False) - math.log(num_batch * self.num_train)) # [8, 128]

        mi_fz   = F.relu(logq_fz - logq_f - logq_z).mean()

        return mi_fz
