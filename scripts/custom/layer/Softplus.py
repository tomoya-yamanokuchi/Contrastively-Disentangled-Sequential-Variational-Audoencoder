from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


class Softplus(nn.Module):
    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super(Softplus, self).__init__()
        self.beta      = beta
        self.threshold = threshold

    def forward(self, input: Tensor) -> Tensor:
        return F.softplus(input, self.beta, self.threshold)