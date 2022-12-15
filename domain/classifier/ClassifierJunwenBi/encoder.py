import torch
import torch.nn as nn
from .conv import conv

class encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]