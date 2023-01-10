import torch
import torch.nn as nn
from omegaconf import OmegaConf
from .SinRegressor import SinRegressor


class EnsembleSinRegressor(nn.Module):
    def __init__(self,
                 network     : OmegaConf,
                 loss        : OmegaConf,
                 num_ensemble: int,
                 **kwargs) -> None:
        super().__init__()
        self.num_ensemble = num_ensemble
        self.model1 = SinRegressor(network, loss)
        self.model2 = SinRegressor(network, loss)
        self.model3 = SinRegressor(network, loss)
        self.model4 = SinRegressor(network, loss)
        self.model5 = SinRegressor(network, loss)


    def forward(self, x):
        mean1, var1 = self.model1.forward(x)
        mean2, var2 = self.model2.forward(x)
        mean3, var3 = self.model3.forward(x)
        mean4, var4 = self.model4.forward(x)
        mean5, var5 = self.model5.forward(x)

        mean = torch.stack((mean1, mean2, mean3, mean4, mean5), axis=0)
        var  = torch.stack(( var1,  var2,  var3,  var4,  var5), axis=0)

        return {
            "mean" : mean,
            "var"  : var,
        }

    def loss_function(self,
                    target                   ,
                    batch_idx                ,
                    results_dict             ,
                    **kwargs) -> dict:

        mean = results_dict["mean"]
        var  = results_dict["var"]

        loss1 = self.model1.loss_function(mean[0], var[0], target)
        loss2 = self.model2.loss_function(mean[1], var[1], target)
        loss3 = self.model3.loss_function(mean[2], var[2], target)
        loss4 = self.model4.loss_function(mean[3], var[3], target)
        loss5 = self.model5.loss_function(mean[4], var[4], target)

        loss = (loss1 + loss2 + loss3 + loss4 + loss5) / self.num_ensemble

        return {
            "loss" : loss
        }