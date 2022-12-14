import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List
import numpy as np
from custom.layer.LinearUnit import LinearUnit
from custom.utility.reparameterize import reparameterize


class LSTMMotionEncoder(nn.Module):
    def __init__(self,
                 lstm_hidden_dim  : int,
                 state_dim        : int,
                 **kwargs) -> None:
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.state_dim       = state_dim

        self.z_rnn    = nn.RNN(self.lstm_hidden_dim * 2, self.lstm_hidden_dim, batch_first=True)
        self.z_mean   = nn.Linear(self.lstm_hidden_dim, state_dim)
        self.z_logvar = nn.Linear(self.lstm_hidden_dim, state_dim)

        # self.forward(Tensor(np.random.randn(32, 8, 512)))
        # import ipdb; ipdb.set_trace()


    def forward(self, lstm_out: Tensor) -> List[Tensor]:
        features, _ = self.z_rnn(lstm_out)
        z_mean      = self.z_mean(features)
        z_logvar    = self.z_logvar(features)
        z_sample    = reparameterize(z_mean, z_logvar, random_sampling=True)
        return z_mean, z_logvar, z_sample



