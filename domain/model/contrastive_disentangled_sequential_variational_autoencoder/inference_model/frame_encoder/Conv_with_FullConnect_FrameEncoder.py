import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List
import numpy as np
import copy
from custom.layer.ConvUnit import ConvUnit
from custom.layer.LinearUnit import LinearUnit

class Conv_with_FullConnect_FrameEncoder(nn.Module):
    def __init__(self,
                 in_channels      : int,
                 conv_out_channels: List[int],
                 conv_fc_out_dims : List[int],
                 **kwargs) -> None:
        super().__init__()

        # ------------ Conv ------------
        modules = nn.ModuleList()
        for out_channels in conv_out_channels:
            modules.append(
                ConvUnit(
                    in_channels,
                    out_channels
                )
            )
            in_channels = out_channels
        self.conv_out         = nn.Sequential(*modules)
        self.summary_conv_out = torchinfo.summary(self.conv_out, input_size=(1, 3, 64, 64))

        # ------------ Linear ------------
        modules = nn.ModuleList()
        in_dim = np.prod(self.summary_conv_out.summary_list[-1].output_size)
        for out_dim in conv_fc_out_dims:
            modules.append(
                LinearUnit(
                    in_dim,
                    out_dim
                )
            )
            in_dim = out_dim
        self.conv_fc = nn.Sequential(*modules)
        self.summary_conv_fc = torchinfo.summary(self.conv_fc, input_size=(1, conv_fc_out_dims[0]))
        return


    def forward(self,  x: Tensor) -> List[Tensor]:
        '''
        - param input: (Tensor) Input tensor to encoder [N x C x H x W]
        -      return: (Tensor) List of latent codes
        ---------------------------------------------------------------------
        The frames are unrolled into the batch dimension for batch processing
        such that x goes from [batch_size, frames, channels, size, size]
                           to [batch_size * frames, channels, size, size]
        '''
        num_batch, step, channle, width, height = x.shape        # 入力データのshapeを取得
        x = x.view(-1, channle, width, height)   # ;print(x.shape) # 最大で4次元データまでなのでreshapeする必要がある
        x = self.conv_out(x)                     # ;print(x.shape) # 畳み込みレイヤで特徴量を取得
        x = torch.flatten(x, start_dim=1)        # ;print(x.shape) # start_dim 以降の次元を flatten
        x = self.conv_fc(x)                      # ;print(x.shape) # 全結合層で特徴量を抽出
        x = x.view(num_batch, step, x.shape[-1]) # ;print(x.shape) # 形状をunrollしてたのを元に戻す(じゃないとLSTMとかに渡せない)
        # import ipdb; ipdb.set_trace()
        return x


