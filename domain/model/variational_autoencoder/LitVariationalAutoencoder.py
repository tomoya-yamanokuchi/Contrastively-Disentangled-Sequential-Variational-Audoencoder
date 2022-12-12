import os
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import numpy as np
import torchinfo
import torch
from torchvision import utils
from torch import Tensor
from torch import optim
from typing import List, Any
import pytorch_lightning as pl
from .VariationalAutoencoder import VariationalAutoencoder
from .. import visualization


class LitVariationalAutoencoder(pl.LightningModule):
    def __init__(self,
                kld_weight,
                **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kld_weight = kld_weight
        self.model = VariationalAutoencoder(**kwargs)
        self.summary = torchinfo.summary(self.model, input_size=(1, 3, 64, 64))


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    def sample_wide_range(self,
                d1_min        : int,
                d1_max        : int,
                d2_min        : int,
                d2_max        : int,
                current_device: int,
                **kwargs) -> Tensor:

        num_sample = 10
        z1      = torch.linspace(start=float(d1_min.to("cpu").numpy()), end=float(d1_max.to("cpu").numpy()), steps=num_sample)
        z2      = torch.linspace(start=float(d2_min.to("cpu").numpy()), end=float(d2_max.to("cpu").numpy()), steps=num_sample)
        grid_z1, grid_z2 = torch.meshgrid(z1, z2, indexing='ij')
        z       = torch.stack([grid_z1.reshape(-1), grid_z2.reshape(-1)], axis=1)
        z       = z.to(current_device)
        samples = self.model.decoder.forward(z)
        return samples


    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results    = self.model.forward(real_img)
        train_loss = self.model.loss_function(
            *results,
            M_N = self.kld_weight, #al_img.shape[0]/ self.num_train_imgs,
            batch_idx = batch_idx
        )
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']


    def validation_step(self, batch, batch_idx):
        real_img, labels = batch
        current_device   = real_img.device
        results          = self.model.forward(real_img)
        recon            = results[0] # Size([num_batch, channel, w, h])
        input            = results[1] # Size([num_batch, channel, w, h])
        mu               = results[2] # Size([num_batch, dim_z])

        val_loss = self.model.loss_function(
            *results,
            M_N = self.kld_weight,
            batch_idx = batch_idx
        )
        self.log("val_loss", val_loss["loss"])

        if pathlib.Path(self.logger.log_dir).exists():
            p = pathlib.Path(self.logger.log_dir + "/reconstruction"); p.mkdir(parents=True, exist_ok=True)
            save_num_recon = 10
            utils.save_image(
                tensor = torch.cat([
                    utils.make_grid(recon[:save_num_recon], nrow=save_num_recon),
                    utils.make_grid(input[:save_num_recon], nrow=save_num_recon),
                ], dim=1),
                fp     = os.path.join(str(p), 'reconstruction_epoch' + str(self.current_epoch)) + '.png',
            )

            p = pathlib.Path(self.logger.log_dir + "/latentn_space"); p.mkdir(parents=True, exist_ok=True)
            visualization.samples(mu, labels,
                save_path = os.path.join(str(p), 'latentn_space' + str(self.current_epoch)) + '.png',
            )

            recon = self.sample_wide_range(mu[:, 0].min(), mu[:, 0].max(), mu[:, 1].min(), mu[:, 1].max(), current_device)
            p     = pathlib.Path(self.logger.log_dir + "/sample_grid"); p.mkdir(parents=True, exist_ok=True)
            utils.save_image(
                tensor = utils.make_grid(recon, nrow=int(np.sqrt(recon.shape[0]))),
                fp     = os.path.join(str(p), 'sample_grid_epoch' + str(self.current_epoch)) + '.png',
            )