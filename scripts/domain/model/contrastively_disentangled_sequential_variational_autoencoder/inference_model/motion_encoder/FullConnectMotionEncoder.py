import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List
import numpy as np
from custom.layer.LinearUnit import LinearUnit
from custom.utility.reparameterize import reparameterize

class FullConnectMotionEncoder(nn.Module):
    def __init__(self,
                 in_dim           : int,
                 hidden_dim       : int,
                 state_dim        : int,
                 **kwargs) -> None:
        super().__init__()

        self.hidden  = LinearUnit(in_dim, hidden_dim)
        self.mean    = nn.Linear(hidden_dim, state_dim)
        self.logvar  = nn.Linear(hidden_dim, state_dim)

        self.summary = torchinfo.summary(self.hidden, input_size=(2, 2048))


    def forward(self,  x: Tensor) -> List[Tensor]:
        num_batch, step, dim = x.shape
        x      = x.view(-1, x.shape[-1])            # shape = [num_batch * step, conv_fc_out_dims[-1]]
        x      = self.hidden(x)                     # shape = [num_batch * step, hidden_dim]
        mean   = self.mean(x)                       # shape = [num_batch * step, state_dim]
        logvar = self.logvar(x)                     # shape = [num_batch * step, state_dim]
        mean   = mean.view(num_batch, step, -1)     # shape = [num_batch, step, state_dim]
        logvar = logvar.view(num_batch, step, -1)   # shape = [num_batch, step, state_dim]
        sample = reparameterize(mean, logvar)       # shape = [num_batch, step, state_dim]
        return mean, logvar, sample



