# -*- coding: utf-8 -*-
import torch
import torchvision
import numpy as np
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from torchvision import utils
from domain.test.TestModel import TestModel
from custom.utility.image_converter import torch2numpy
import cv2; cv2.namedWindow('img', cv2.WINDOW_NORMAL)


# log = "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221208211521]"
log = "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221210035007]-[remote_3090]-32219"
log = "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221212212525]-[remote_3090]-momo"
log = "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221212235346]-[dl-box]-nene"

# ----------------------------------------------------------------------------------
model   = "cdsvae4"
log_dir = "/hdd_mount/logs_cdsvae/{}/".format(model)
test    = TestModel(
    config_dir  = log_dir + log,
    checkpoints = "last.ckpt"
)
device     = test.device
model      = test.load_model()
dataloader = test.load_dataloader()
# ----------------------------------------------------------------------------------
step       = 25
num_slice  = 1
_step      = 0

for index, img_dict in dataloader:
    img = img_dict["images"]

    for test_index in range(len(img)):
    # for test_index in range(1):
        print("[{}-{}] - [{}/{}]".format(index.min(), index.max(), test_index+1, len(img)))

        num_batch, step = img.shape[:2]
        for m in range(num_batch):
            x = img[m].unsqueeze(0)
            (f_mean, f_logvar, f_sample), (z_mean, z_logvar, z_sample) = model.encode(x)

            # 1系列の画像に対してcontentを変化させる
            for i in range(30):
                x_sample = model.forward_fixed_motion(z_mean, f_mean, f_logvar)
                x_sample = torchvision.utils.make_grid(x_sample, nrow=step, padding=0, pad_value=0.0, normalize=True)
                x_sample = torch2numpy(x_sample)
                x_sample = x_sample[_step]
                x_sample = cv2.cvtColor(x_sample, cv2.COLOR_RGB2BGR)
                cv2.imshow("img", x_sample)
                cv2.waitKey(200)
