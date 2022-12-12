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
from domain.visualize.vector_heatmap import VectorHeatmap


test = TestModel(
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_205",
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_202",
    config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_235",
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_232",
    checkpoints = "last.ckpt"
)
device     = test.device
model      = test.load_model()
dataloader = test.load_dataloader()


iter_dataloader = iter(dataloader)


# 1回読み込み
index, batch = next(iter_dataloader)
img_seq1     = batch[0].unsqueeze(dim=0).to(device)
# 2回読み込み
index, batch = next(iter_dataloader)
index, batch = next(iter_dataloader)
img_seq2     = batch[7].unsqueeze(dim=0).to(device)


return_dict_seq1 = model(img_seq1)
return_dict_seq2 = model(img_seq2)

x_recon1 = model.decode(return_dict_seq1["z_mean"], return_dict_seq1["f_mean"])
x_recon2 = model.decode(return_dict_seq2["z_mean"], return_dict_seq2["f_mean"])

x_recon_z1_f2 = model.decode(return_dict_seq1["z_mean"], return_dict_seq2["f_mean"])
x_recon_z2_f1 = model.decode(return_dict_seq2["z_mean"], return_dict_seq1["f_mean"])


# ---------------------------------
z1  = return_dict_seq1["z_mean"].to("cpu").numpy()
z2  = return_dict_seq2["z_mean"].to("cpu").numpy()
# import ipdb; ipdb.set_trace()
vectorHeatmap = VectorHeatmap()
vectorHeatmap.pause_show(np.concatenate((z1[0].transpose(), z2[0].transpose()), axis=1) , interval=-1)
# ---------------------------------

# import ipdb; ipdb.set_trace()



save_sequence = 1
step          = 8 # sprite
# step          = 25 # valve
num_slice     = 1
images        = []
for n in range(save_sequence):
    images.append(utils.make_grid(x_recon1[n][::num_slice], nrow=step))
    images.append(utils.make_grid(img_seq1[n][::num_slice], nrow=step))
    images.append(utils.make_grid(x_recon2[n][::num_slice], nrow=step))
    images.append(utils.make_grid(img_seq2[n][::num_slice], nrow=step))

    images.append(utils.make_grid(torch.ones_like(img_seq2[n])[::num_slice], nrow=step))

    images.append(utils.make_grid(x_recon_z1_f2[n][::num_slice], nrow=step))
    images.append(utils.make_grid(x_recon_z2_f1[n][::num_slice], nrow=step))


    # 入力画像と再構成画像を並べて保存
    utils.save_image(
        tensor = torch.cat(images, dim=1),
        fp     = "/home/tomoya-y/workspace/pytorch_lightning_VAE/fig/feature_swapping_{}.png".format(test.config_dir.split("/")[-1]),
    )