import os
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import numpy as np
import json
from pprint import pprint
import torch
from torch import optim
from torchvision import utils
from torch import Tensor
from typing import List, Any
import pytorch_lightning as pl
from .ContrastiveDisentangledSequentialVariationalAutoencoder import ContrastiveDisentangledSequentialVariationalAutoencoder
from .scheduler.SchedulerFactory import SchedulerFactory
from ..save_numpy import save_as_numpy_scalar



class LitContrastiveDisentangledSequentialVariationalAutoencoder(pl.LightningModule):
    def __init__(self,
                 config,
                 num_train) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config       = config
        self.num_train    = num_train
        self.model        = ContrastiveDisentangledSequentialVariationalAutoencoder(**config.model, num_train=num_train)
        self.summary_dict = None
        # self.summary = torchinfo.summary(self.model, input_size=(131, 8, 3, 64, 64))


    def forward(self, input, **kwargs) -> Any:
        return self.model.forward(input)

    def encode(self, img):
        return self.model.encode(img)

    def decode(self, z, f):
        return self.model.decode(z, f)

    def forward_fixed_motion_for_classification(self, img):
        return self.model.forward_fixed_motion_for_classification(img)

    def forward_fixed_motion(self, *args):
        return self.model.forward_fixed_motion(*args)


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
        index, data          = batch  # shape = [num_batch, step, channel, w, h], Eg.) [128, 8, 3, 64, 64])
        # assert len(img_tuple) == 3
        # print("batch_idx: {} = {}".format(batch_idx, index[0]))
        # import ipdb; ipdb.set_trace()

        x, label_A, label_D, c_aug, m_aug = data['images'], data['A_label'], data['D_label'], data['c_aug'], data['m_aug']

        results_dict              = self.model.forward(x)     # original
        results_dict_aug_context  = self.model.forward(c_aug) # augment context
        results_dict_aug_dynamics = self.model.forward(m_aug) # augment dynamics

        loss = self.model.loss_function(
            x                         = x,
            batch_idx                 = batch_idx,
            results_dict              = results_dict,
            results_dict_aug_context  = results_dict_aug_context,
            results_dict_aug_dynamics = results_dict_aug_dynamics,
        )

        # if self.summary_dict is None:
        #     for key in loss.keys():
        #         self.summary_dict[key] = []
        #     for key in self.summary_dict.keys():
        #         self.summary_dict[key].append(loss[key])
        #     save_as_numpy_scalar(loss, self.logger.log_dir)
        # else:
        #     for key in self.summary_dict.keys():
        #         self.summary_dict[key].append(loss[key])
        #     save_as_numpy_scalar(loss, self.logger.log_dir)

        self.log("index_0", index[0])
        self.log_dict({key: val.item() for key, val in loss.items()}, sync_dist=True)
        return loss['loss']



    def validation_step(self, batch, batch_idx):
        index, data          = batch  # shape = [num_batch, step, channel, w, h], Eg.) [128, 8, 3, 64, 64])
        # assert len(img_tuple) == 3
        x, label_A, label_D, c_aug, m_aug = data['images'], data['A_label'], data['D_label'], data['c_aug'], data['m_aug']

        results_dict              = self.model.forward(x)     # original
        results_dict_aug_context  = self.model.forward(c_aug) # augment context
        results_dict_aug_dynamics = self.model.forward(m_aug) # augment dynamics

        loss = self.model.loss_function(
            x                         = x,
            batch_idx                 = batch_idx,
            results_dict              = results_dict,
            results_dict_aug_context  = results_dict_aug_context,
            results_dict_aug_dynamics = results_dict_aug_dynamics,
        )
        self.log("val_loss", loss["loss"])
        if batch_idx == 0:
            self.save_progress(
                # *img_tuple,
                *(x, c_aug, m_aug),
                results_dict,
            )


    def save_progress(self,
                      img_batch,
                      img_aug_context,
                      img_aug_dynamics,
                      results_dict: dict,
                      name_tag: str=""):

        if pathlib.Path(self.logger.log_dir).exists():
            p = pathlib.Path(self.logger.log_dir + "/reconstruction"); p.mkdir(parents=True, exist_ok=True)
            num_batch, step, channel, width, height = img_batch.shape

            save_sequence = 9 # np.minimum(10, mod)
            images        = []
            for n in range(save_sequence):
                images_unit = []
                images_unit.append(utils.make_grid(torch.ones_like(img_batch[n]), nrow=step, padding=2, pad_value=1.0, normalize=False))
                images_unit.append(utils.make_grid(results_dict["x_recon"][n],    nrow=step, padding=2, pad_value=0.0, normalize=True))
                images_unit.append(utils.make_grid(              img_batch[n],    nrow=step, padding=2, pad_value=0.0, normalize=True))
                images_unit.append(utils.make_grid(        img_aug_context[n],    nrow=step, padding=2, pad_value=0.0, normalize=True))
                images_unit.append(utils.make_grid(       img_aug_dynamics[n],    nrow=step, padding=2, pad_value=0.0, normalize=True))
                images.append(torch.cat(images_unit, dim=1))

            print("\n\n---------------------------------------")
            print(" [img_batch] min. max = [{}, {}]".format(img_batch[1].min(), img_batch[1].max()))
            print(" [  images ] min. max = [{}, {}]".format(   images[1].min(),    images[1].max()))
            print("---------------------------------------\n\n")

            # save input and reconstructed images
            '''
                Plese check if range of img is [0.0, 1.0].
                Because utils.save_image() assums that tensor image is in range [0.0, 1.0] internally.
            '''
            # import ipdb; ipdb.set_trace()
            utils.save_image(
                # tensor = torch.cat(images, dim=2),
                tensor = torch.cat(torch.chunk(torch.cat(images, dim=2), chunks=3, dim=-1), dim=1),
                fp     = os.path.join(str(p), 'reconstruction_epoch' + str(self.current_epoch)) + name_tag + '.png',
            )
