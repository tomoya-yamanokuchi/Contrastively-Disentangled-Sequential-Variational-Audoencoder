# -*- coding: utf-8 -*-
from omegaconf import OmegaConf
import torch
import torchvision
import numpy as np
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from torchvision import utils
from domain.test.TestModel import TestModel
from domain.datamodule.DataModuleFactory import DataModuleFactory
from custom.utility.image_converter import torch2numpy
import cv2; cv2.namedWindow('img', cv2.WINDOW_NORMAL)


config     = OmegaConf.load("./conf/datamodule/sprite_JunwenBi.yaml")
datamodule = DataModuleFactory().create(**config)
datamodule.setup(stage="test")
dataloader = datamodule.test_dataloader()


for index, img_dict in dataloader:
    # img = img_dict["images"]
    img = img_dict["m_aug"]

    for test_index in range(len(img)):
        print("[{}-{}] - [{}/{}]".format(index.min(), index.max(), test_index+1, len(img)))

        num_batch, step = img.shape[:2]
        for m in range(num_batch):
            print(test_index, m)
            img_1seq = img[m]

            x_sample = torchvision.utils.make_grid(img_1seq, nrow=step, padding=0, pad_value=0.0, normalize=True)
            x_sample = torch2numpy(x_sample)
            x_sample = cv2.cvtColor(x_sample, cv2.COLOR_RGB2BGR)

            cv2.imshow("img", x_sample)
            cv2.waitKey(300)
            # import ipdb; ipdb.set_trace()
