from torch import nn


class UpConvEnd(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvEnd, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels    = in_channels,
                out_channels   = out_channels,
                kernel_size    = 4,
                stride         = 2,
                padding        = 1,
            ),
            # nn.Tanh()
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
