import copy
import numpy as np
import torch
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.test.TestModel import TestModel
from custom.visualize.VectorHeatmap import VectorHeatmap
from custom.utility.image_converter import torch2numpy


log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221122171135]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221122183834]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221122233949]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221122233946]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221123204704]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=128]-[dim_z=8]-[1000epoch]-[20221124031555]"

log = "[c-dsvae]-[sprite_aug]-[dim_f=128]-[dim_z=8]-[500epoch]-[20221125102327]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=72]-[dim_z=7]-[500epoch]-[20221127005444]"

log = "[c-dsvae]-[sprite_aug]-[dim_f=72]-[dim_z=7]-[500epoch]-[20221127035954]"
# ----------------------------------------------------------------------------------
model   = "C-DSVAE2"
log_dir = "/home/tomoya-y/workspace/pytorch_lightning_VAE/logs/{}/".format(model)
test    = TestModel(
    config_dir  = log_dir + log,
    checkpoints = "last.ckpt"
)
device     = test.device
model      = test.load_model()
dataloader = test.load_dataloader()
# ----------------------------------------------------------------------------------
import cv2
# cv2.namedWindow('img', cv2.WINDOW_NORMAL)

vectorHeatmap = VectorHeatmap()
for index, img_tuple in dataloader:
    (img, img_aug_context, img_aug_dynamics) = img_tuple
    f = []
    for test_index in range(len(img)):
        print("[{}-{}] - [{}/{}]".format(index.min(), index.max(), test_index+1, len(img_tuple[0])))

        img_seq        = img[test_index].unsqueeze(dim=0).to(device)
        return_dict    = model(img_seq)
        _f             = return_dict["f_mean"].to("cpu").numpy()
        _, dim_f       = _f.shape
        f.append(copy.deepcopy(_f))

    # import ipdb; ipdb.set_trace()
    images = torch.concat(torch.split(img, 1, dim=0), dim=3).squeeze(0)
    images = torch.concat(torch.split(images, 1, dim=0), dim=-1).squeeze(0)
    images = torch2numpy(images)
    images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    # cv2.imshow("ddd", images)
    # cv2.waitKey(10)
    # import ipdb; ipdb.set_trace()
    vectorHeatmap.pause_show(np.concatenate(f, axis=0), interval=1)

