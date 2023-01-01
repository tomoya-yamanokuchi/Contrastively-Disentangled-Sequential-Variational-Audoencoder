from ast import Lambda
from re import M
import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List, Any
import numpy as np
from .Reshape import Reshape


class Decoder(nn.Module):
    def __init__(self,
                 encoder_summary  : Any,
                 hidden_dims      : List[int],
                 latent_dim       : int,
                 **kwargs) -> None:
        super().__init__()

        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim, np.prod(encoder_summary.summary_list[-1].output_size)),
                Reshape(-1, *encoder_summary.summary_list[-1].output_size[1:]),
            )
        )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels    = hidden_dims[i],
                        out_channels   = hidden_dims[i + 1],
                        kernel_size    = 3,
                        stride         = 2,
                        padding        = 1,
                        output_padding = 1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels    = hidden_dims[-1],
                    out_channels   = hidden_dims[-1],
                    kernel_size    = 3,
                    stride         = 2,
                    padding        = 1,
                    output_padding = 1
                ),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels  = hidden_dims[-1],
                    out_channels = 3,
                    kernel_size  = 3,
                    padding      = 1
                ),
                nn.Tanh()
            )
        )
        self.decoder = nn.Sequential(*modules)
        self.summary = torchinfo.summary(self.decoder, input_size=(1, latent_dim))



    def forward(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        : return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        return result
