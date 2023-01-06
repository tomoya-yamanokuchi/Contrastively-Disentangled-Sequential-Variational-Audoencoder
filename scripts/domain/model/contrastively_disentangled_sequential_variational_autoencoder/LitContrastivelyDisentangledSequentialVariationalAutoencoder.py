import os
import pathlib
import torch
from torch import optim
from torchvision import utils
from typing import List, Any
import pytorch_lightning as pl
from .ContrastivelyDisentangledSequentialVariationalAutoencoder import ContrastivelyDisentangledSequentialVariationalAutoencoder
from .scheduler.SchedulerFactory import SchedulerFactory

'''
    setting to fix random seed
'''
# import random
# def torch_fix_seed(seed=42):
#     # Python random
#     random.seed(seed)
#     # Numpy
#     np.random.seed(seed)
#     # Pytorch
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.use_deterministic_algorithms(True)
# torch_fix_seed()



class LitContrastivelyDisentangledSequentialVariationalAutoencoder(pl.LightningModule):
    def __init__(self,
                 config,
                 num_train) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config       = config
        self.num_train    = num_train
        self.model        = ContrastivelyDisentangledSequentialVariationalAutoencoder(
            **config.model,  loss=config.loss, num_train=num_train
        )
        self.summary_dict = None


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

    def forward_fixed_content(self, *args):
        return self.model.forward_fixed_content(*args)

    def sample_motion(self, *args):
        return self.model.motion_prior.sample(*args)

    def sample_content(self, *args):
        return self.model.content_prior.sample(*args)


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
        index, data = batch
        x           = data['images']
        c_aug       = data['c_aug']
        m_aug       = data['m_aug']

        results_dict              = self.model.forward(x)
        results_dict_aug_context  = self.model.forward(c_aug)
        results_dict_aug_dynamics = self.model.forward(m_aug)

        loss = self.model.loss_function(
            x                         = x,
            batch_idx                 = batch_idx,
            results_dict              = results_dict,
            results_dict_aug_context  = results_dict_aug_context,
            results_dict_aug_dynamics = results_dict_aug_dynamics,
        )
        self.log("index_0", index[0])
        self.log_dict({key: val.item() for key, val in loss.items()}, sync_dist=True)
        return loss['loss']


    def validation_step(self, batch, batch_idx):
        index, data = batch
        x           = data['images']
        c_aug       = data['c_aug']
        m_aug       = data['m_aug']

        results_dict              = self.model.forward(x)
        results_dict_aug_context  = self.model.forward(c_aug)
        results_dict_aug_dynamics = self.model.forward(m_aug)

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
                images_unit.append(utils.make_grid(torch.ones_like(img_batch[n]),     nrow=step, padding=2, pad_value=1.0, normalize=False))
                images_unit.append(utils.make_grid(results_dict["x_recon"][n],        nrow=step, padding=2, pad_value=0.0, normalize=True))
                images_unit.append(utils.make_grid(              img_batch[n],        nrow=step, padding=2, pad_value=0.0, normalize=True))
                images_unit.append(utils.make_grid(        img_aug_context[n],        nrow=step, padding=2, pad_value=0.0, normalize=True))
                images_unit.append(utils.make_grid(img_batch[n] - img_aug_context[n], nrow=step, padding=2, pad_value=0.0, normalize=False))
                images_unit.append(utils.make_grid(       img_aug_dynamics[n],        nrow=step, padding=2, pad_value=0.0, normalize=True))
                images.append(torch.cat(images_unit, dim=1))

            print("\n\n---------------------------------------")
            print(" [img_batch] min. max = [{}, {}]".format(img_batch[1].min(), img_batch[1].max()))
            print(" [  images ] min. max = [{}, {}]".format(   images[1].min(),    images[1].max()))
            print("---------------------------------------\n\n")

            '''
                Plese check if range of img is [0.0, 1.0].
                Because utils.save_image() assums that tensor image is in range [0.0, 1.0] internally.
            '''
            utils.save_image(
                tensor = torch.cat(torch.chunk(torch.cat(images, dim=2), chunks=3, dim=-1), dim=1),
                fp     = os.path.join(str(p), 'reconstruction_epoch' + str(self.current_epoch)) + name_tag + '.png',
            )
