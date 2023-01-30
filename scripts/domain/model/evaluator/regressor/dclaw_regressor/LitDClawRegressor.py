import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import pytorch_lightning as pl
from torch import optim, tensor
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from torchvision import utils
from .EnsembleDClawRegressor import EnsembleDClawRegressor
from custom.scheduler.SchedulerFactory import SchedulerFactory
from custom.utility.to_numpy import to_numpy
from domain.visualize.DClawDataPlot import DClawDataPlot



class LitDClawRegressor(pl.LightningModule):
    def __init__(self,
                 config    : OmegaConf,
                 num_train : OmegaConf,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config       = config
        self.num_train    = num_train
        self.num_ensemble = config.num_ensemble
        self.model        = EnsembleDClawRegressor(**config.model, loss=config.loss, num_ensemble=config.num_ensemble)
        self.summary_dict = None
        # self.summary = torchinfo.summary(self.model, input_size=(131, 8, 3, 64, 64))


    def forward(self, *args):
        return self.model.forward(*args)


    def configure_optimizers(self):
        optimizer = optim.Adam(
            params = self.parameters(),
            lr     = self.config.optimizer.lr,
            betas  = tuple(self.config.optimizer.betas)
        )
        scheduler = SchedulerFactory().create(**self.config.scheduler, optimizer=optimizer, max_epochs=self.config.trainer.max_epochs)
        if scheduler is None: return optimizer
        else                : return [optimizer,], [scheduler,]


    def training_step(self, batch, batch_idx):
        index, data = batch  # shape = [num_batch, step, channel, w, h], Eg.) [128, 8, 3, 64, 64])
        x = data['images']
        y = data['state']

        # import ipdb; ipdb.set_trace()
        results_dict = self.model.forward(x)

        loss = self.model.loss_function(
            target       = y,
            batch_idx    = batch_idx,
            results_dict = results_dict,
        )

        self.log("index_0", index[0])
        self.log_dict({key: val.item() for key, val in loss.items()}, sync_dist=True)
        return loss['loss']


    def validation_step(self, batch, batch_idx):
        index, data = batch  # shape = [num_batch, step, channel, w, h], Eg.) [128, 8, 3, 64, 64])
        x = data['images']
        y = data['state']

        # import ipdb; ipdb.set_trace()
        results_dict = self.model.forward(x)

        loss = self.model.loss_function(
            target       = y,
            batch_idx    = batch_idx,
            results_dict = results_dict,
        )
        self.log("val_loss", loss["loss"])

        # q, mod = np.divmod(self.current_epoch+1, 5)
        # if (mod==0) or (self.current_epoch==0):
        if batch_idx == 0:
            self.save_progress(
                (x, y),
                results_dict,
            )


    def save_progress(self,
                      xy,
                      results_dict,
                      name_tag: str=""):

        if pathlib.Path(self.logger.log_dir).exists():


            (images, y) = xy
            num_batch, step, dim_y = y.shape
            x = np.linspace(1, step, step)

            y             = to_numpy(y)
            mean_ensemble = to_numpy(results_dict["mean"])
            var_ensemble  = to_numpy(results_dict["var"])

            # << get predictive distribution >>
            mean = mean_ensemble.mean(axis=0)
            var  = (var_ensemble + mean_ensemble**2).mean(axis=0) - mean**2

            for index in range(num_batch)[:3]:
                p = pathlib.Path(self.logger.log_dir + "/regression/sequence_{}".format(index)); p.mkdir(parents=True, exist_ok=True)
                # p = pathlib.Path(self.logger.log_dir + "/regression"); p.mkdir(parents=True, exist_ok=True)

                # << plot data: total mean and variance >>
                sin_plot = DClawDataPlot(xlabel="x", ylabel="y", title="y=sin(x)")
                sin_plot.plot_true_function(x, y[index])
                sin_plot.plot_prediction(x, mean[index], np.sqrt(var[index]))
                sin_plot.save_fig(os.path.join(str(p), 'mean_std_epoch' + str(self.current_epoch)) + name_tag + '.png')

                # << plot data: indivisdual mean and variance >>
                sin_plot = DClawDataPlot(xlabel="x", ylabel="y", title="y=sin(x)")
                sin_plot.plot_true_function(x, y[index])
                for m in range(self.num_ensemble):
                    sin_plot.plot_prediction(x, mean_ensemble[m][index], np.sqrt(var_ensemble[m][index]))
                sin_plot.save_fig(os.path.join(str(p), 'individual_mean_std_epoch' + str(self.current_epoch)) + name_tag + '.png')

                # << save image sequence >>
                image = utils.make_grid(images[index], nrow=step, padding=2, pad_value=0.0, normalize=True)
                utils.save_image(image, fp=os.path.join(str(p), 'observation_epoch' + str(self.current_epoch)) + name_tag + '.png')