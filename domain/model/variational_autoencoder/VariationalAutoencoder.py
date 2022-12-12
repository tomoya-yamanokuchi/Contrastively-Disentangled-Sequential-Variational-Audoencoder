import torch
from torch import nn
from torch import Tensor
from typing import List
from .Encoder import Encoder
from .Decoder import Decoder
from torch.nn import functional as F


class VariationalAutoencoder(nn.Module):
    def __init__(self,
                 in_channels      : int,
                 conv_out_channels: int,
                 latent_dim       : List = None,
                 **kwargs) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, conv_out_channels, latent_dim)
        self.decoder = Decoder(self.encoder.summary, conv_out_channels, latent_dim)


    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encoder.forward(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decoder(z), input, mu, log_var]


    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons      = args[0]
        input       = args[1]
        mu          = args[2]
        log_var     = args[3]

        kld_weight  = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss    = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {
            'loss': loss,
            'Reconstruction_Loss':recons_loss.detach(),
            'KLD' :-kld_loss.detach(),
            # 'betaKLD' :
        }