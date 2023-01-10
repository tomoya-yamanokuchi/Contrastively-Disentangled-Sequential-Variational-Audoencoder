import torch
import torch.nn as nn
from omegaconf import OmegaConf
from custom.layer.Softplus import Softplus


class DClawRegressor(nn.Module):
    def __init__(self,
                 dim_in    : int,
                 dim_hidden: int,
                 dim_out   : int,
                 loss      : OmegaConf,
                 **kwargs) -> None:
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(True),
            nn.Linear(dim_hidden, dim_out),
        )

        self.var = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(True),
            nn.Linear(dim_hidden, dim_out),
            Softplus(),
        )
        self.beta    = loss.beta


    def forward(self, x):
        # << mean >>
        mean    = self.mean(x)
        # << var >>
        var_min = torch.Tensor([1e-8]).type_as(x)
        var_max = torch.Tensor([100]).type_as(x)
        var     = self.var(x) + var_min
        var     = torch.clamp(var, min=var_min, max=var_max)
        return mean, var


    def loss_function(self,
                    mean,
                    var,
                    target,
                    **kwargs) -> dict:

        beta_nll_loss = self.beta_nll_loss(
            mean     = mean,
            variance = var,
            target   = target,
        )
        return beta_nll_loss


    def beta_nll_loss(self, mean, variance, target):
        """Compute beta-NLL loss

        :param mean    : Predicted mean of shape B x D
        :param variance: Predicted variance of shape B x D
        :param target  : Target of shape B x D
        :param beta    : Parameter from range [0, 1] controlling relative
            weighting between data points, where `0` corresponds to
            high weight on low error points and `1` to an equal weighting.

        :returns       : Loss per batch element of shape B
        """
        loss = 0.5 * (((target - mean) ** 2 / variance) + variance.log())

        if self.beta > 0:
            loss = loss * (variance.detach() ** self.beta)

        # import ipdb; ipdb.set_trace()
        return loss.sum(axis=-1).mean()