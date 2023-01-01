from torch import nn

class ConvEnd(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvEnd, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = 4,
                stride       = 1,
                padding      = 0,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)