from torch import nn

class LinearUnit(nn.Module):
    def __init__(self, in_dim, out_dim, batchnorm=True):
        super(LinearUnit, self).__init__()
        if batchnorm:
            self.model = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(0.2),
            )

        '''
        context ベクトルの f_mean, f_logvaeを求めるのに，
        LeakyReLUはあっていいのか？ f ~ N(mu, sigma)　なのに
        '''


    def forward(self, x):
        return self.model(x)