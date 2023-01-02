import torch
from torch import Tensor
from torch import nn
from typing import List
from custom.layer.LinearUnit import LinearUnit
from custom.utility.reparameterize import reparameterize


class ContextEncoder(nn.Module):
    def __init__(self,
                 lstm_hidden_dim  : List[int],
                 context_dim      : int,
                 **kwargs) -> None:
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.f_mean          = LinearUnit(self.lstm_hidden_dim * 2, context_dim, False)
        self.f_logvar        = LinearUnit(self.lstm_hidden_dim * 2, context_dim, False)


    def forward(self, lstm_out: Tensor) -> List[Tensor]:
        '''
        num_batch, step, dim = lstm_out.shape
        The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        '''
        backward   = lstm_out[:, 0, self.lstm_hidden_dim:2 * self.lstm_hidden_dim]
        frontal    = lstm_out[:, -1, 0:self.lstm_hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        f_mean     = self.f_mean(lstm_out_f)
        f_logvar   = self.f_logvar(lstm_out_f)
        f_sample   = reparameterize(f_mean, f_logvar, random_sampling=True)
        return f_mean, f_logvar, f_sample