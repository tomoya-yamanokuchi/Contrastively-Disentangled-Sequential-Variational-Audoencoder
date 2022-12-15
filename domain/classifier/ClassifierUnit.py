from torch import nn



class ClassifierUnit(nn.Module):
    def __init__(self, lstm_hidden_dim, dim_class):
        super(ClassifierUnit, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.ReLU(True),
            nn.Linear(lstm_hidden_dim, dim_class)
        )

    def forward(self, x):
        return self.model(x)