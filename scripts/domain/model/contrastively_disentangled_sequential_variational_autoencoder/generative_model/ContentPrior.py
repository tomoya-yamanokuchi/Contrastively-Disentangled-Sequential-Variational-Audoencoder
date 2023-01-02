import torch
from custom import reparameterize


class ContentPrior:
    def __init__(self, context_dim):
        self.context_dim = context_dim

    def mean(self, num_batch: int):
        return torch.zeros(num_batch, self.context_dim).cuda()

    def logvar(self, num_batch: int):
        return torch.zeros(num_batch, self.context_dim).cuda()

    def sample(self, num_batch: int):
        mean   = self.mean(num_batch)
        logvar = self.logvar(num_batch)
        sample = reparameterize(mean, logvar)
        return mean, logvar, sample
