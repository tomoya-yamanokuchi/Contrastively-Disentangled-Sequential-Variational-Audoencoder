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
from custom import to_numpy

config     = OmegaConf.load("./conf/datamodule/sprite_JunwenBai.yaml")
datamodule = DataModuleFactory().create(**config)
datamodule.setup(stage="test")
dataloader = datamodule.test_dataloader()


for index, img_dict in dataloader:
    img     = img_dict["images"]
    D_label = img_dict["D_label"]

    D_label    = np.array(to_numpy(D_label), dtype=np.int)
    num_motion = len(set(D_label))
    assert num_motion == 9
    for d_label in range(num_motion):
        img_grouped           = img[D_label==d_label]
        num_grouped_img, step = img_grouped.shape[:2]

        # import ipdb; ipdb.set_trace()
        for i in range(num_grouped_img):
            print("(d_label={}) {}/{}".format(d_label, i+1, num_grouped_img))

            img_1seq = img_grouped[i]
            x_sample = torchvision.utils.make_grid(img_1seq, nrow=step, padding=0, pad_value=0.0, normalize=True)
            x_sample = torch2numpy(x_sample)
            x_sample = cv2.cvtColor(x_sample, cv2.COLOR_RGB2BGR)

            cv2.imshow("img", x_sample)
            cv2.waitKey(300)
            # import ipdb; ipdb.set_trace()
