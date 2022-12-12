import os
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import numpy as np
import json
from pprint import pprint
import torchinfo
import torch
from torch import optim
from torchvision import utils
from torch import Tensor
from typing import List, Any
import pytorch_lightning as pl
from .ContrastiveDisentangledSequentialVariationalAutoencoder import ContrastiveDisentangledSequentialVariationalAutoencoder
from .scheduler.SchedulerFactory import SchedulerFactory
from .. import visualization

import cv2
from custom.utility.image_converter import torch2numpy
from custom.utility.reoder import reorder


def print_log(print_string, log=None, verbose=True):
    if verbose:
        print("{}".format(print_string))
    if log is not None:
        log = open(log, 'a')
        log.write('{}\n'.format(print_string))
        log.close()


class LitContrastiveDisentangledSequentialVariationalAutoencoder(pl.LightningModule):
    def __init__(self,
                 config,
                 num_train) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config    = config
        self.num_train = num_train
        self.model     = ContrastiveDisentangledSequentialVariationalAutoencoder(**config.model, num_train=num_train)
        # self.summary = torchinfo.summary(self.model, input_size=(131, 8, 3, 64, 64))

        print_log(self.model, None)
        print_log(vars(config), None)


    def forward(self, input, **kwargs) -> Any:
        return self.model.forward(input)


    def decode(self, z, f):
        '''
        input:
            - z: shape = []
            - f: shape = []
        '''
        # import ipdb; ipdb.set_trace()

        num_batch, step, _ = z.shape
        # z         = z.view(num_batch, step, -1)
        # import ipdb; ipdb.set_trace()
        # f         = f.view(num_batch, step, -1)
        x_recon   = self.model.frame_decoder(torch.cat((z, f.unsqueeze(1).expand(num_batch, step, -1)), dim=2))
        return x_recon


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
        # import ipdb; ipdb.set_trace()

        # x, label_A, label_D, c_aug, m_aug = reorder(data['images']), data['A_label'], data['D_label'], reorder(data['c_aug']), reorder(data['m_aug'])
        # x, label_A, label_D, c_aug, m_aug = x.cuda(), label_A.cuda(), label_D.cuda(), c_aug.cuda(), m_aug.cuda()
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

        # self.save_progress(img_batch, results_dict)
        self.log("val_loss", loss["loss"])
        self.log_dict({key: val.item() for key, val in loss.items()}, sync_dist=True)
        return loss['loss']



    def validation_step(self, batch, batch_idx):
        index, data          = batch  # shape = [num_batch, step, channel, w, h], Eg.) [128, 8, 3, 64, 64])
        # assert len(img_tuple) == 3

        # x, label_A, label_D, c_aug, m_aug = reorder(data['images']), data['A_label'], data['D_label'], reorder(data['c_aug']), reorder(data['m_aug'])
        # x, label_A, label_D, c_aug, m_aug = x.cuda(), label_A.cuda(), label_D.cuda(), c_aug.cuda(), m_aug.cuda()
        x, label_A, label_D, c_aug, m_aug = data['images'], data['A_label'], data['D_label'], data['c_aug'], data['m_aug']
        # import ipdb; ipdb.set_trace()

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
