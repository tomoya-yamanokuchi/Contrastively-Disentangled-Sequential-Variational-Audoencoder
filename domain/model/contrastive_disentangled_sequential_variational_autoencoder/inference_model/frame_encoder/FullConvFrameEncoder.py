import numpy as np
from torch import Tensor
from torch import nn
from custom.layer.ConvUnit import ConvUnit
from custom.layer.ConvEnd import ConvEnd

'''
Such encoder is a convolutional neural network with 5 layers of channels [64, 128, 256, 512, 128].
'''

class FullConvFrameEncoder(nn.Module):
    def __init__(self,
                 in_channels      : int,
                 dim_frame_feature: int,
                 **kwargs) -> None:
        super().__init__()

        self.dim_frame_feature = dim_frame_feature

        nf      = 64
        self.c1 = ConvUnit(in_channels, nf)                 # input size: (in_channels) x 64 x 64
        self.c2 = ConvUnit(nf, nf * 2)                      # state size:          (nf) x 32 x 32
        self.c3 = ConvUnit(nf * 2, nf * 4)                  # state size:        (nf*2) x 16 x 16
        self.c4 = ConvUnit(nf * 4, nf * 8)                  # state size:        (nf*4) x  8 x  8
        self.c5 = ConvEnd(nf * 8, self.dim_frame_feature)   # state size:        (nf*8) x  4 x  4 --> dim_frame_feature x 1 x 1

        # self.forward(Tensor(np.random.randn(32, 8, 3, 64, 64)))
        # import ipdb; ipdb.set_trace()


    def forward(self, x: Tensor):
        num_batch, step, channle, width, height = x.shape
        x  = x.view(-1, channle, width, height)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(num_batch, step, self.dim_frame_feature)




