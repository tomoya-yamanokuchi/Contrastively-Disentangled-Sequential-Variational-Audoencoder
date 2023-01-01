import torch
from omegaconf import OmegaConf
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from ..model.ModelFactory import ModelFactory
from ..datamodule.DataModuleFactory import DataModuleFactory


class TestModel:
    def __init__(self, config_dir: str, checkpoints: str):
        self.config_dir         = config_dir
        self.checkpoints        = checkpoints
        self.config             = OmegaConf.load(config_dir + "/config.yaml")
        self.config.reload.path = config_dir + "/checkpoints/{}".format(checkpoints)

    def load_model(self):
        lit_model_class = ModelFactory().create(self.config.model.name)
        lit_model       = lit_model_class.load_from_checkpoint(self.config.reload.path)
        lit_model.freeze()
        # print(lit_model.model.state_dict()["frame_decoder.deconv_fc.0.model.1.weight"])
        return lit_model.eval().cuda()

    def load_dataloader(self, stage="test"):
        datamodule = DataModuleFactory().create(**self.config.datamodule)
        datamodule.setup(stage=stage)
        return datamodule.test_dataloader()