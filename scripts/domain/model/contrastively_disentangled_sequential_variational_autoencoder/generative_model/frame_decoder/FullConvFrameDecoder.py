from torch import Tensor
from torch import nn
from typing import List
from .upconv.UpConvFirst import UpConvFirst
from .upconv.UpConvUnit import UpConvUnit
from .upconv.UpConvEnd import UpConvEnd


class FullConvFrameDecoder(nn.Module):
    def __init__(self,
                 in_dim      : int,
                 out_channels : int,
                 **kwargs) -> None:
        super().__init__()
        self.in_dim = in_dim
        nf          = 64
        self.upc1   = UpConvFirst(in_dim, nf * 8)
        self.upc2   = UpConvUnit(nf * 8, nf * 4)  # state size. (nf*8) x  4 x  4
        self.upc3   = UpConvUnit(nf * 4, nf * 2)  # state size. (nf*4) x  8 x  8
        self.upc4   = UpConvUnit(nf * 2, nf)      # state size. (nf*2) x 16 x 16
        self.upc5   = UpConvEnd(nf, out_channels) # state size.   (nf) x 32 x 32


    def forward(self,  x: Tensor) -> List[Tensor]:
        num_batch, step, dim = x.shape
        d1     = self.upc1(x.view(-1, self.in_dim, 1, 1))
        d2     = self.upc2(d1)
        d3     = self.upc3(d2)
        d4     = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(num_batch, step, output.shape[1], output.shape[2], output.shape[3])
        return output