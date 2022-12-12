import torch


class ContextPrior:
    def __init__(self, context_dim):
        self.context_dim = context_dim

    def mean(self, x):
        return torch.zeros(self.context_dim).type_as(x)

    def logvar(self, x):
        # 後でexp()の処理が入るのでここでは0
        return torch.zeros(self.context_dim).type_as(x)

    def dist(self, x):
        return self.mean(x), self.logvar(x)
