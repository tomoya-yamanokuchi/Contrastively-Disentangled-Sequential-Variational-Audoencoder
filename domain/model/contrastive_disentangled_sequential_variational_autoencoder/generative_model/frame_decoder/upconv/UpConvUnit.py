from torch import nn


class UpConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvUnit, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels    = in_channels,
                out_channels   = out_channels,
                kernel_size    = 4,
                stride         = 2,
                padding        = 1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)
