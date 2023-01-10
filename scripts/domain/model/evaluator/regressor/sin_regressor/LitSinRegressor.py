import os
import sys; import pathlib
from cv2 import randn; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
from torch import optim
from .SinRegressor import SinRegressor
from .EnsembleSinRegressor import EnsembleSinRegressor
from custom.scheduler.SchedulerFactory import SchedulerFactory
from custom.utility.to_numpy import to_numpy
from domain.visualize.SinDataPlot import SinDataPlot


class LitSinRegressor(pl.LightningModule):
    def __init__(self,
                 config    : OmegaConf,
                 num_train : OmegaConf,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config       = config
        self.num_train    = num_train
        self.num_ensemble = config.num_ensemble
        self.model        = EnsembleSinRegressor(**config.model, loss=config.loss, num_ensemble=config.num_ensemble)
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
        x = data['input']
        y = data['output']

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
        x      = data['input']
        y      = data['output']
        x_true = data['input_true'][0]  # 真のデータなので1バッチ分でいい
        y_true = data['output_true'][0] # 真のデータなので1バッチ分でいい

        # import ipdb; ipdb.set_trace()

        results_dict = self.model.forward(x_true)

        loss = self.model.loss_function(
            target       = y_true,
            batch_idx    = batch_idx,
            results_dict = results_dict,
        )
        self.log("val_loss", loss["loss"])

        # import ipdb; ipdb.set_trace()
        # print("batch_idx = ", batch_idx)
        # print("self.current_epoch = ", self.current_epoch)

        q, mod = np.divmod(self.current_epoch+1, 10)
        if (mod==0) or (self.current_epoch==0):
            self.save_progress(
                (x, y),
                (x_true, y_true),
                results_dict,
            )


    def save_progress(self,
                      xy,
                      xy_true,
                      results_dict,
                      name_tag: str=""):

        if pathlib.Path(self.logger.log_dir).exists():
            p         = pathlib.Path(self.logger.log_dir + "/regression"); p.mkdir(parents=True, exist_ok=True)
            file_path = os.path.join(str(p), 'sin_prediction_epoch' + str(self.current_epoch)) + name_tag + '.png'

            (x, y)           = xy
            (x_true, y_true) = xy_true

            x      = to_numpy(x).reshape(-1)
            y      = to_numpy(y).reshape(-1)
            x_true = to_numpy(x_true).reshape(-1)
            y_true = to_numpy(y_true).reshape(-1)

            mean_ensemble = to_numpy(results_dict["mean"]).squeeze(-1)
            var_ensemble  = to_numpy(results_dict["var"]).squeeze(-1)
            # print("std = ", np.sqrt(var))
            # import ipdb; ipdb.set_trace()
            mean = mean_ensemble.mean(axis=0)
            var  = (var_ensemble + mean_ensemble**2).mean(axis=0) - mean**2

            sin_plot = SinDataPlot(xlabel="x", ylabel="y", title="y=sin(x)", yminmax=(-1.5, 1.5))
            sin_plot.plot_observation(x, y)
            sin_plot.plot_true_function(x_true, y_true)
            sin_plot.plot_prediction(x_true, mean, np.sqrt(var))
            sin_plot.save_fig(file_path)


            file_path = os.path.join(str(p), 'sin_prediction_ensemble_epoch' + str(self.current_epoch)) + name_tag + '.png'
            sin_plot = SinDataPlot(xlabel="x", ylabel="y", title="y=sin(x)", yminmax=(-1.5, 1.5))
            sin_plot.plot_observation(x, y)
            sin_plot.plot_true_function(x_true, y_true)
            for m in range(self.num_ensemble):
                sin_plot.plot_prediction(x_true, mean_ensemble[m], np.sqrt(var_ensemble[m]))
            sin_plot.save_fig(file_path)