import os, time
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, omegaconf
# ------------------------------------
import sys; import pathlib; p=pathlib.Path("./"); sys.path.append(str(p.parent.resolve()))
from domain.datamodule.DataModuleFactory import DataModuleFactory
from domain.model.ModelFactory import ModelFactory
from domain.test.TestModel import TestModel
from domain.visualize.save_plot import plot_2D_latent_space
from domain.visualize.MotionVectorPlot import MotionVectorPlot
from custom import to_numpy, save_image, save_image_as_gif
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class TestDClaw:
    def load_model(self, group, model):
        self.log_dir = "/hdd_mount/logs_cdsvae/{}/{}".format(group, model)
        test         = TestModel(config_dir=self.log_dir, checkpoints="last.ckpt")
        self.model, self.config_model = test.load_model()
        self._make_test_dir()


    def _make_test_dir(self):
        self.save_dir = os.path.join(self.log_dir, 'test/{}'.format(time.time()))
        os.makedirs(self.save_dir, exist_ok=True)


    def load_evaluation_dataset(self):
        datamodule = DataModuleFactory().create(**self.config_model.datamodule)
        datamodule.setup(stage="test")
        self.test_dataloader = datamodule.test_dataloader()


    def evaluate(self, num_action:int):
        '''
            num_action: number of action variation
        '''
        for index, data in self.test_dataloader:
            x             = data['images']
            f_sample_list = []
            num_content   = int(len(index)/num_action)
            for i in range(num_content):
                x_inter_domain   = x[(i*num_action):((i+1)*num_action)].contiguous()
                return_dict      = self.model(x_inter_domain)
                x_recon          = return_dict["x_recon"]
                f_sample         = return_dict["f_sample"]

                save_image(save_path=os.path.join(test.save_dir, "content{}_x.png"      .format(i)), image=x_inter_domain,      normalize=True)
                f_sample_list.append(f_sample)
            break

        return f_sample_list



if __name__ == '__main__':


    # model_cdsvae = "[c-dsvae]-[robel_dclaw_deterministic]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230128155146]-[remote_3090]-unique_content_s"
    model_cdsvae = "[c-dsvae]-[robel_dclaw_deterministic]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230128124230]-[remote_tsukumo3090ti]-unique_content_s"


    group_model  = "cdsvae_dclaw_deterministic"
    test = TestDClaw()
    test.load_model(group=group_model, model=model_cdsvae)
    test.load_evaluation_dataset()

    num_action = 6
    f_encoded_mean  = test.evaluate(num_action=num_action, test_index=0)
    num_test, dim_f = f_encoded_mean

