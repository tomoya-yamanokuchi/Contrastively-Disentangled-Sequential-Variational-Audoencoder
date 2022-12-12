import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List
import numpy as np

class Encoder(nn.Module):
    def __init__(self,
                 in_channels      : int,
                 conv_out_channels: List[int],
                 latent_dim       : int,
                 **kwargs) -> None:
        super().__init__()

        modules = []
        for out_channels in conv_out_channels:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels  = in_channels,
                        out_channels = out_channels,
                        kernel_size  = 3,
                        stride       = 2,
                        padding      = 1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU()
                )
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*modules)
        self.summary = torchinfo.summary(self.encoder, input_size=(1, 3, 64, 64))

        self.fc_mu   = nn.Linear(np.prod(self.summary.summary_list[-1].output_size), latent_dim)
        self.fc_var  = nn.Linear(np.prod(self.summary.summary_list[-1].output_size), latent_dim)


    def forward(self,  input: Tensor) -> List[Tensor]:
        """
        - param input: (Tensor) Input tensor to encoder [N x C x H x W]
        -      return: (Tensor) List of latent codes
        """
        result  = self.encoder(input)                #;print(result.shape)
        result  = torch.flatten(result, start_dim=1) #;print(result.shape)

        mu      = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


