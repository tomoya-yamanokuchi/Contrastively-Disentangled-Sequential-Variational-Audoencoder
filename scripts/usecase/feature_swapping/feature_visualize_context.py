import time
import os
import numpy as np
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.test.TestModel import TestModel
from custom.visualize.VectorHeatmap import VectorHeatmap
from custom.utility.image_converter import torch2numpy
from domain.datamodule.DataModuleFactory import DataModuleFactory


log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221122171135]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221122183834]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221122233949]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221122233946]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=14]-[dim_z=7]-[500epoch]-[20221123204704]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=128]-[dim_z=8]-[1000epoch]-[20221124031555]"

log = "[c-dsvae]-[sprite_aug]-[dim_f=128]-[dim_z=8]-[500epoch]-[20221125102327]"
log = "[c-dsvae]-[sprite_aug]-[dim_f=72]-[dim_z=7]-[500epoch]-[20221127005444]"

log = "[c-dsvae]-[sprite_aug]-[dim_f=72]-[dim_z=7]-[500epoch]-[20221127035954]"


# cdsvae
model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103062258]-melco_mmm'
model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103071526]-melco_mmm'
model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103080755]-melco_mmm'
model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103090028]-melco_mmm'
model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103095302]-melco_mmm'

# DSVAE
# model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103062313]-remote3090_mmm'
# model = '[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230103071925]-remote3090_mmm'
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
datamodule = DataModuleFactory().create(**test.config.datamodule)
datamodule.setup(stage="test")
dataloader = datamodule.grouped_dataloader()
# ----------------------------------------------------------------------------------
# import cv2
# cv2.namedWindow('img', cv2.WINDOW_NORMAL)

dirname  = time.time()
save_dir = "./vectorHeatmap/{}".format(dirname)
os.makedirs(save_dir, exist_ok=True)

for index, img_dict in dataloader:
    image           = img_dict["images"]
    num_batch, step = image.shape[:2]

    for test_index in range(num_batch):
        print("[{}-{}] - [{}/{}]".format(index.min(), index.max(), test_index+1, num_batch))

        (f_mean, f_logvar, f_sample), (z_mean, z_logvar, z_sample) = model.encode(image)

        f = f_mean.to("cpu").numpy()
        z = z_mean.to("cpu").numpy()

        VectorHeatmap().save_plot(v=np.transpose(f), save_path="{}/content_text_index_{}".format(save_dir, test_index))
        [VectorHeatmap().save_plot(v=np.transpose(z[t]), save_path="{}/motion_text_index_{}_step_{}".format(save_dir, test_index, t)) for t in range(step)]
        sys.exit()

