import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List
import numpy as np
from custom.layer.LinearUnit import LinearUnit
from custom.utility.reparameterize import reparameterize



class BiLSTMEncoder(nn.Module):
    def __init__(self,
                 in_dim    : int,
                 hidden_dim: List[int],
                 **kwargs) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # ------------ LSTM ------------
        self.lstm_out = nn.LSTM(
            input_size    = in_dim,
            hidden_size   = hidden_dim,
            num_layers    = 1,
            bidirectional = True, # if True:  output dim is lstm_hidden_dim * 2
            batch_first   = True,
        )
        # x = self.forward(Tensor(np.random.randn(32, 8, 128)))
        # import ipdb; ipdb.set_trace()


    def forward(self,  x: Tensor) -> List[Tensor]:
        '''
        num_batch, step, dim = lstm_out.shape
        The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        '''
        # num_batch, step, dim  = x.shape
        lstm_out, _ = self.lstm_out(x) # shape=[num_batch, step, lstm_hidden_dim*2]
        return lstm_out