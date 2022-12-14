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

for index, img_dict in dataloader:
    img = img_dict["images"]

    for test_index in range(len(img)):
    # for test_index in range(1):
        print("[{}-{}] - [{}/{}]".format(index.min(), index.max(), test_index+1, len(img)))

        recon_x_sample, recon_x = model.forward_fixed_motion_for_classification(img)
        num_batch, step, channel, width, height = recon_x.shape

        for m in range(num_batch):
            flattened = []
            for x in [recon_x_sample[m], recon_x[m]]:
                x_flat = torchvision.utils.make_grid(x, nrow=step, padding=0, pad_value=0.0, normalize=True)
                x_flat = torch2numpy(x_flat)
                flattened.append(x_flat)

            flattened = np.concatenate(flattened, axis=0)
            flattened = cv2.cvtColor(flattened, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", flattened)
            cv2.waitKey(1000)

            # import ipdb; ipdb.set_trace()

