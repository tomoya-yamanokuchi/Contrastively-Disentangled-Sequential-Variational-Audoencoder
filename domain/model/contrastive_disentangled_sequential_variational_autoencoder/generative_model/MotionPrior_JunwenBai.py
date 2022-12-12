import torch
from torch import Tensor, device
from torch import nn
from typing import List, Any
from custom import reparameterize


class MotionPrior_JunwenBai(nn.Module):
    def __init__(self,
                 state_dim  : int,
                 hidden_dim : int,
                 **kwargs) -> None:
        super().__init__()
        self.state_dim        = state_dim
        self.hidden_dim       = hidden_dim
        self.z_prior_lstm_ly1 = nn.LSTMCell(state_dim, hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(hidden_dim, hidden_dim)

        self.z_prior_mean     = nn.Linear(hidden_dim, state_dim)
        self.z_prior_logvar   = nn.Linear(hidden_dim, state_dim)



    def forward(self, z_post) -> Tensor:
        z_out      = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means    = None
        z_logvars  = None
        batch_size, step, dim_z = z_post.shape

        z_t     = torch.zeros(batch_size, dim_z).type_as(z_post)
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).type_as(z_post)
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).type_as(z_post)
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).type_as(z_post)
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).type_as(z_post)

        for i in range(step):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t   = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior    = reparameterize(z_mean_t, z_logvar_t)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out     = z_prior.unsqueeze(1)
                z_means   = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out     = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means   = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_post[:,i,:]
        return z_means, z_logvars, z_out
