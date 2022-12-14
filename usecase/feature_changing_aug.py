# -*- coding: utf-8 -*-
import torch
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from torchvision import utils
from domain.test.TestModel import TestModel
from custom.utility.image_converter import torch2numpy

import cv2
cv2.namedWindow('img', cv2.WINDOW_NORMAL)



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

    '''
    "images" : data_ancher,
    "c_aug"  : c_aug_anchor,
    "m_aug"  : m_aug_anchor,
    "A_label": A_label_ancher,
    "D_label": D_label_ancher,
    "index"  : index
    '''
    img = img_dict["images"]

    for test_index in range(len(img)):
    # for test_index in range(1):
        print("[{}-{}] - [{}/{}]".format(index.min(), index.max(), test_index+1, len(img)))
        # import ipdb; ipdb.set_trace()
        img_seq         = img[test_index].unsqueeze(dim=0).to(device)
        return_dict_seq = model(img_seq)
        # import ipdb; ipdb.set_trace()
        x_recon         = model.decode(return_dict_seq["z_mean"], return_dict_seq["f_mean"])

        z = return_dict_seq["z_mean"]
        f = return_dict_seq["f_mean"]

        for m in range(50):
            x_recon = model.decode(z, f)
            image = utils.make_grid(x_recon[0][0], nrow=1, normalize=True)
            image = torch2numpy(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", image)
            cv2.waitKey(50)

            '''
            dynamic info
            '''
            z += torch.randn_like(z) * 0.5
            # ========= 232 =============
            # import ipdb; ipdb.set_trace()
            # z[:, :, :3] += torch.randn_like(z[:, :, :3])*2.0
            # z[:, :, 0] += torch.randn_like(z[:, :, 0])*2.0
            # z[:, :, 2] += torch.randn_like(z[:, :, 1])*2.0
            # z[:, :, 4:6] += torch.randn_like(z[:, :, 4:6])*2.0
            # z[:, :, 3] += torch.randn_like(z[:, :, 3])*2.0
            # z[:, :, 6:] += torch.randn_like(z[:, :, 6:])*2.0
            '''
            color info
            '''
            # z[:, :, 0:6]  += torch.randn_like(z[:, :, 0:6])
            # z[:, :, 8:12] += torch.randn_like(z[:, :, 8:12])
            # z[:, :, 13:]  += torch.randn_like(z[:, :, 13:])

            # f += torch.randn_like(f) * 0.25
