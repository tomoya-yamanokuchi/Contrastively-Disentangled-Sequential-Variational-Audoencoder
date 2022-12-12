import sys; import pathlib
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.model.ModelFactory import ModelFactory
from domain.datamodule.DataModuleFactory import DataModuleFactory
from domain.test.TestModel import TestModel
from custom.utility.image_converter import torch2numpy
import os
from torchvision import utils

import cv2
import numpy as np

import cv2
cv2.namedWindow('img', cv2.WINDOW_NORMAL)

test = TestModel(
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_232",
    # config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_235",
    config_dir  = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/DSVAE/version_306",
    checkpoints = "last.ckpt"
)
device     = test.device
model      = test.load_model()
dataloader = test.load_dataloader()

step       = 25
num_slice  = 1

for index, img_batch in dataloader:
    # import ipdb; ipdb.set_trace()
    for test_index in range(len(img_batch)):
    # for test_index in range(1):
        print("[{}-{}] - [{}/{}]".format(index.min(), index.max(), test_index+1, len(img_batch)))

        img_seq         = img_batch[test_index].unsqueeze(dim=0).to(device)
        return_dict_seq = model(img_seq)
        x_recon         = model.decode(return_dict_seq["z_mean"], return_dict_seq["f_mean"])

        z = return_dict_seq["z_mean"]
        f = return_dict_seq["f_mean"]

        for m in range(50):
            x_recon = model.decode(z, f)
            img     = utils.make_grid(x_recon[0][0], nrow=1, normalize=True)
            img     = torch2numpy(img)
            img     = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", img)
            cv2.waitKey(50)

            '''
            dynamic info
            '''
            # ========= 232 =============
            # z[:, :, 6] += torch.randn_like(z[:, :, 6])*2.0
            # z[:, :, 7] += torch.randn_like(z[:, :, 7])*2.0
            # z[:, :, 12] += torch.randn_like(z[:, :, 12])*2.0

            # ========= 235 =============
            # z[:, :, 3] += torch.randn_like(z[:, :, 6])*2.0
            # z[:, :, 4] += torch.randn_like(z[:, :, 7])*2.0
            # z[:, :, 6] += torch.randn_like(z[:, :, 12])*2.0

            '''
            color info
            '''
            # z[:, :, 0:6]  += torch.randn_like(z[:, :, 0:6])
            # z[:, :, 8:12] += torch.randn_like(z[:, :, 8:12])
            # z[:, :, 13:]  += torch.randn_like(z[:, :, 13:])

            # f += torch.randn_like(f)
            # f[:, 10:30] += torch.randn_like(f[:, 10:30])
            # f[:, :30] += torch.randn_like(f[:, :30])
            # f[:, 31:43] += torch.randn_like(f[:, 31:43])
            # f[:, 49:] += torch.randn_like(f[:, 49:])

            # ========= 235 =============
            f += torch.randn_like(f) * 0.5
            # f[:, 1] += torch.randn_like(f[:, 1])
            # f[:, 2] += torch.randn_like(f[:, 2])
            # f[:, 6] += torch.randn_like(f[:, 6]) # <  >
            # f[:, 7] += torch.randn_like(f[:, 7]) # <  >
            # f[:, 8] += torch.randn_like(f[:, 8]) # < hair color >
            # f[:, 50:] += torch.randn_like(f[:, 50:]) # < hair color >
            # f[:, 13] += torch.randn_like(f[:, 13])
            # f[:, 21] += torch.randn_like(f[:, 21])
            # f[:, 23] += torch.randn_like(f[:, 23])
            # f[:, 47] += torch.randn_like(f[:, 47])
            # f[:, 59] += torch.randn_like(f[:, 59]) # < hair color >
            # f[:, 61] += torch.randn_like(f[:, 61]) #