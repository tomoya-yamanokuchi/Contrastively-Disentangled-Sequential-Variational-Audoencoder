from torch import nn

'''
A block consisting of
    - transposed convolution
    - batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
'''

class ConvUnitTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnitTranspose, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels    = in_channels,
                out_channels   = out_channels,
                kernel_size    = kernel_size,
                stride         = stride,
                padding        = padding,
                output_padding = output_padding,
            ),
            nn.BatchNorm2d(out_channels),
            nonlinearity,
        )

    def forward(self, x):
        return self.model(x)