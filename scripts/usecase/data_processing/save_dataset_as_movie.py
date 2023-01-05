import torch, os
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.datamodule.DataModuleFactory import DataModuleFactory
from torchvision.io import write_video
from custom.utility.image_converter import torch2numpy
from omegaconf import OmegaConf

name              = "sprite_JunwenBai"
config_datamodule = OmegaConf.load("./conf/datamodule/{}.yaml".format(name))
# -----------------------------------------
config_datamodule.config_dataloader.train.batch_size = 10
config_datamodule.config_dataloader.train.shuffle    = False
# -----------------------------------------
datamodule        = DataModuleFactory().create(**config_datamodule)
datamodule.setup(stage="fit")
dataloader        = datamodule.train_dataloader()

iter_dataloader = iter(dataloader)
index, batch,   = next(iter_dataloader)
assert index[0] == 0

index, batch,   = next(iter_dataloader)
index, batch,   = next(iter_dataloader)

images = batch["images"]
num_batch, step, channel, width, height = images.shape

# import ipdb; ipdb.set_trace()

save_dir = "gif/" + name
os.makedirs(save_dir, exist_ok=True)
for i in range(num_batch):
    print("[{}/{}]".format(i, num_batch))
    img_seq = images[i]
    img_seq = torch2numpy(img_seq)
    write_video(
        filename    = save_dir + "/{}_num_batch{}.mp4".format(name, i),
        video_array = img_seq,
        fps         = 10.0,
    )

