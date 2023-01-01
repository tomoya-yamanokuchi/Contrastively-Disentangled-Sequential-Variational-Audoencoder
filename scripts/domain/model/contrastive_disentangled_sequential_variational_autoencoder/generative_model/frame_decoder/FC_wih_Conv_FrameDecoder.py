from re import I
import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List
import numpy as np
import copy
from custom.layer.ConvUnit import ConvUnit
from custom.layer.ConvUnitTranspose import ConvUnitTranspose
from custom.layer.LinearUnit import LinearUnit


class FC_wih_Conv_FrameDecoder(nn.Module):
    def __init__(self,
                 in_dim      : int,
                 deconv_fc_out_dims : List[int],
                 deconv_in_channel  : int,
                 deconv_out_channels: List[int],
                 **kwargs) -> None:
        super().__init__()

        assert np.divmod(deconv_fc_out_dims[-1], np.sqrt(deconv_fc_out_dims[-1]))[-1] == 0.0 # 2D shapeに変更できるか確認
        self.sqrt_final_fc_dim = int(np.sqrt(deconv_fc_out_dims[-1]))
        self.init_in_dim = copy.deepcopy(in_dim)

        # ------------ Linear ------------
        modules = nn.ModuleList()
        for out_dim in deconv_fc_out_dims:
            modules.append(
                LinearUnit(
                    in_dim,
                    out_dim
                )
            )
            in_dim = out_dim
        self.deconv_fc         = nn.Sequential(*modules)
        self.summary_deconv_fc = torchinfo.summary(self.deconv_fc, input_size=(1, self.init_in_dim))

        # ------------ Conv ------------
        # 全結合層の出力次元とdeconvの入力チャンネルからdeconvへの入力サイズ(w, h)を計算
        self.deconv_in_channel = deconv_in_channel
        quotient, remainder    = np.divmod(deconv_fc_out_dims[-1], deconv_in_channel)
        assert remainder == 0
        quotient, remainder   = np.divmod(quotient, np.sqrt(quotient))
        assert remainder == 0.0
        self.width_deconv_in  = int(quotient)
        self.height_deconv_in = int(quotient)

        in_channels = copy.deepcopy(deconv_in_channel)

        modules = nn.ModuleList()
        for out_channels in deconv_out_channels:
            modules.append(
                ConvUnitTranspose(
                    in_channels    = in_channels,
                    out_channels   = out_channels,
                    kernel_size    = 5,
                    stride         = 2,
                    padding        = 2,
                    output_padding = 1,
                )
            )
            in_channels = out_channels
        # last layer to reconstruct image
        modules.append(
            ConvUnitTranspose(
                in_channels    = in_channels,
                out_channels   = 3, # because outout is RGB image
                kernel_size    = 5,
                stride         = 1,
                padding        = 2,
                output_padding = 0,
                nonlinearity   = nn.Tanh(),
            )
        )
        self.deconv_out                     = nn.Sequential(*modules)
        self.summary_deconv_out             = torchinfo.summary(self.deconv_out, input_size=(1, deconv_in_channel, self.width_deconv_in, self.height_deconv_in))
        self.width_recon, self.height_recon = self.summary_deconv_out.summary_list[-1].output_size[-2:]


    def forward(self,  x: Tensor) -> List[Tensor]:
        num_batch, step, dim = x.shape
        x = x.view(-1, dim)                                                                 # [num_batch * step, dim]
        x = self.deconv_fc(x)                                                               # [num_batch * step, deconv_fc_out_dims[-1]]
        x = x.view(-1, self.deconv_in_channel, self.width_deconv_in, self.height_deconv_in) # [num_batch * step, ~]
        x = self.deconv_out(x)                                                              # [num_batch * step, rgb_channel, image_width, image_height]
        return x.view(num_batch, step, 3, self.width_recon, self.height_recon)
