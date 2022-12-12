import copy
import numpy as np
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.test.TestModel import TestModel
from custom.visualize.VectorHeatmap import VectorHeatmap


log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221122171135]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221122175122]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221122233949]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221122233946]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221123204704]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221123201940]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=128]-[dim_z=8]-[1000epoch]-[20221124031555]"

log = "[c-dsvae]-[sprite_aug]-[dim_f=128]-[dim_z=8]-[500epoch]-[20221125102327]"

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

vectorHeatmap = VectorHeatmap()
for index, img_tuple in dataloader:
    (img, img_aug_context, img_aug_dynamics) = img_tuple
    z = []
    for test_index in range(len(img)):
        print("[{}-{}] - [{}/{}]".format(index.min(), index.max(), test_index+1, len(img_tuple[0])))

        img_seq        = img[test_index].unsqueeze(dim=0).to(device)
        return_dict    = model(img_seq)
        _z             = return_dict["z_mean"].to("cpu").numpy()
        _, step, dim_z = _z.shape
        z.append(copy.deepcopy(_z.reshape(step, dim_z).transpose()))

        vectorHeatmap.pause_show(_z.reshape(step, dim_z).transpose(), interval=0.05)