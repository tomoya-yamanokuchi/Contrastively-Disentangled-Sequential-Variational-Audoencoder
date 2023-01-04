# -*- coding: utf-8 -*-
import os
import sys
import cv2
import time
import torch
import torchvision
import numpy as np
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from torchvision import utils
from domain.test.TestModel import TestModel
from custom.utility.image_converter import torch2numpy
from custom.utility.create_gif import create_gif
# cv2.namedWindow('img', cv2.WINDOW_NORMAL)


# cdsvae
model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103062258]-melco_mmm'
model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103071526]-melco_mmm'
model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103080755]-melco_mmm'
model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103090028]-melco_mmm'
model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103095302]-melco_mmm'

# DSVAE
# model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103062313]-remote3090_mmm'
model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103071925]-remote3090_mmm'
# model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103081559]-remote3090_mmm'
# model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103091218]-remote3090_mmm'
# model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103100839]-remote3090_mmm'

group = 'cdsvae_sprite'

# ----------------------------------------------------------------------------------
log_dir = "/hdd_mount/logs_cdsvae/{}/".format(group)
test    = TestModel(
    config_dir  = log_dir + model,
    checkpoints = "last.ckpt"
)
model      = test.load_model()
dataloader = test.load_dataloader()
# ----------------------------------------------------------------------------------
num_slice  = 1
_step      = 0

# fixed = "motion"
fixed = "content"

save_dir   = "./gif/{}".format(time.time())
os.makedirs(save_dir, exist_ok=True)
num_save   = 5
# ----------------------------------------------------------------------------------
for index, img_dict in dataloader:
    img = img_dict["images"] # ([128, 8, 3, 64, 64])
    for test_index in range(len(img)):
        print("[{}-{}] - [{}/{}]".format(index.min(), index.max(), test_index+1, len(img)))

        num_batch, step = img.shape[:2]
        for m in range(num_batch)[:num_save]:
            print("index_batch = ", m)
            x = img[m].unsqueeze(0)
            (f_mean, f_logvar, f_sample), (z_mean, z_logvar, z_sample) = model.encode(x)

            images = [] # 保存用

            # 1系列の画像に対して変化させる
            for i in range(100):
            # for i in range(1):
                if fixed == "motion"  :
                    x_sample = model.forward_fixed_motion(z_mean)
                    x_sample = x_sample[0][::num_slice]

                    if True:
                        x_sample = x_sample[4] # 特定のステップだけを取り出す
                        x_sample = torchvision.utils.make_grid(x_sample, nrow=1, padding=0, pad_value=0.0, normalize=True)
                    else:
                        x_sample = torchvision.utils.make_grid(x_sample, nrow=step, padding=0, pad_value=0.0, normalize=True)

                    x_sample = torch2numpy(x_sample)
                    x_sample = cv2.cvtColor(x_sample, cv2.COLOR_RGB2BGR)
                    images.append(x_sample)

                elif fixed == "content" :
                    x_sample = model.forward_fixed_content(f_mean, step)
                    x_sample = x_sample[0]
                    for t in range(step):
                        _x = torchvision.utils.make_grid(x_sample[t], nrow=1, padding=0, pad_value=0.0, normalize=True)
                        _x = torch2numpy(_x)
                        _x = cv2.cvtColor(_x, cv2.COLOR_RGB2BGR)
                        images.append(_x)
                    images.append(np.zeros_like(_x))
            create_gif(
                images   = images,
                fname    = "{}/fixed_{}_{}_{}.gif".format(save_dir, fixed, test_index, m),
                duration = 200 if fixed == "motion" else 100,
            )
        sys.exit()