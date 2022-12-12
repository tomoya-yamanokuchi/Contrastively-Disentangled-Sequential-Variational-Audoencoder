import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.model.ModelFactory import ModelFactory
from domain.datamodule.DataModuleFactory import DataModuleFactory
from domain.test.TestModel import TestModel
from custom.utility import image_converter
import os
from torchvision import utils

import cv2
import numpy as np
from custom.visualize.VectorHeatmap import VectorHeatmap



log = "[c-dsvae]-[sprite_aug]-[dim_f=72]-[dim_z=7]-[500epoch]-[20221127005444]"

log = "[c-dsvae]-[sprite_aug]-[dim_f=72]-[dim_z=7]-[500epoch]-[20221127035954]"
# ----------------------------------------------------------------------------------
model   = "C-DSVAE2"
log_dir = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/{}/".format(model)
test    = TestModel(
    config_dir  = log_dir + log,
    checkpoints = "last.ckpt"
    # checkpoints = "epoch=.ckpt"
)
device     = test.device
model      = test.load_model()
dataloader = test.load_dataloader()
# ----------------------------------------------------------------------------------


iter_dataloader = iter(dataloader)


# 1回読み込み
index, (img, img_aug_context, img_aug_dynamics) = next(iter_dataloader)
img_seq1 = img[0].unsqueeze(dim=0).to(device)

# 2回読み込み
index, (img, img_aug_context, img_aug_dynamics) = next(iter_dataloader)
img_seq2 = img[8].unsqueeze(dim=0).to(device)


return_dict_seq1 = model(img_seq1)
return_dict_seq2 = model(img_seq2)

x_recon1 = model.decode(return_dict_seq1["z_mean"], return_dict_seq1["f_mean"])
x_recon2 = model.decode(return_dict_seq2["z_mean"], return_dict_seq2["f_mean"])

x_recon_z2_f1 = model.decode(return_dict_seq2["z_mean"], return_dict_seq1["f_mean"])
x_recon_z1_f2 = model.decode(return_dict_seq1["z_mean"], return_dict_seq2["f_mean"])

# ---------------------------------
z1  = return_dict_seq1["z_mean"].to("cpu").numpy()
z2  = return_dict_seq2["z_mean"].to("cpu").numpy()
# ---------------------------------

# import ipdb; ipdb.set_trace()



save_sequence = 1
step          = 8 # sprite
# step          = 25 # valve
num_slice     = 1
images        = []
for n in range(save_sequence):
    images.append(utils.make_grid(x_recon1[n][::num_slice], nrow=step, padding=2, pad_value=0.0, normalize=True))
    images.append(utils.make_grid(img_seq1[n][::num_slice], nrow=step, padding=2, pad_value=0.0, normalize=True))
    images.append(utils.make_grid(x_recon2[n][::num_slice], nrow=step, padding=2, pad_value=0.0, normalize=True))
    images.append(utils.make_grid(img_seq2[n][::num_slice], nrow=step, padding=2, pad_value=0.0, normalize=True))

    images.append(utils.make_grid(torch.ones_like(img_seq2[n])[::num_slice], nrow=step, padding=2, pad_value=1.0, normalize=False))

    images.append(utils.make_grid(x_recon_z2_f1[n][::num_slice], nrow=step, padding=2, pad_value=0.0, normalize=True))
    images.append(utils.make_grid(x_recon_z1_f2[n][::num_slice], nrow=step, padding=2, pad_value=0.0, normalize=True))



    # 入力画像と再構成画像を並べて保存
    utils.save_image(
        tensor = torch.cat(images, dim=1),
        fp     = "/home/tomoya-y/workspace/pytorch_lightning_VAE/fig/feature_swapping_C-DSVAE{}.png".format(test.config_dir.split("/")[-1]),
    )