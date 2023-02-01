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
from domain.visualize.SensorPlot import SensorPlot
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


    def evaluate(self):
        index_motion1        = 0
        index_motion2        = 1
        num_use_index        = 2
        num_action_variation = 6
        for index, data in self.test_dataloader:
            x         = data['images'] # image
            x_motion1 = x[index_motion1::num_action_variation][:num_use_index].contiguous()
            x_motion2 = x[index_motion2::num_action_variation][:num_use_index].contiguous()

            # import ipdb; ipdb.set_trace()

            z_sample_encoded_list = []
            for i, _x in enumerate([x_motion1, x_motion2]):
                # << encode original data >>
                return_dict      = self.model(_x)
                z_sample_encoded = return_dict["z_sample"]
                x_recon          = return_dict["x_recon"]

                save_image(save_path=os.path.join(test.save_dir, "motion{}_x.png"      .format(i)), image=_x,      normalize=True)
                save_image(save_path=os.path.join(test.save_dir, "motion{}_x_recon.png".format(i)), image=x_recon, normalize=True)

                for j in range(num_use_index):
                    _x_listed = [np.transpose(np.array(_xit.squeeze(0)*255, dtype=np.uint8), (1, 2, 0)) for _xit in np.split(to_numpy(_x[j]), _x[j].shape[0], axis=0)]
                    save_image_as_gif(images=_x_listed, fname=os.path.join(test.save_dir, "x_motion_{}_content{}_.gif".format(i, j)), duration=100)

                z_sample_encoded_list.append(z_sample_encoded)
            break

        return z_sample_encoded_list



if __name__ == '__main__':



    model_cdsvae = "[c-dsvae]-[action_norm_valve]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230119151208]-[remote_tsukumo3090ti]-popo"
    model_cdsvae = "[c-dsvae]-[action_norm_valve]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230123053236]-[remote_tsukumo3090ti]-conv_content"
    model_cdsvae = "[c-dsvae]-[action_norm_valve]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230126024438]-[remote_tsukumo3090ti]-kkk"
    model_cdsvae = "[c-dsvae]-[action_norm_valve]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230126024440]-[remote_3090]-kkk"
    model_cdsvae = "[c-dsvae]-[action_norm_valve]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230126090633]-[remote_tsukumo3090ti]-vvv"

    model_cdsvae = "[c-dsvae]-[robel_dclaw_deterministic]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230127190220]-[remote_tsukumo3090ti]-mmm"

    model_cdsvae = "[c-dsvae]-[robel_dclaw_deterministic]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230128155146]-[remote_3090]-unique_content_s"
    # model_cdsvae = "[c-dsvae]-[robel_dclaw_deterministic]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230128124230]-[remote_tsukumo3090ti]-unique_content_s"
    model_cdsvae = "[c-dsvae_high_dim]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230201073453]-[remote_tsukumo3090ti]-high_dim"

    # group_model  = "cdsvae_dclaw_deterministic"
    group_model  = "cdsvae_sprite_revisit"
    test = TestDClaw()
    test.load_model(group=group_model, model=model_cdsvae)
    test.load_evaluation_dataset()

    z_mean_encoded_2motion_list   = test.evaluate()
    num_test_1motion, step, dim_z = z_mean_encoded_2motion_list[0].shape

    # ---- t-SNE learning ----
    tsne       = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=25, n_iter=1000)
    z_embedded = tsne.fit_transform(to_numpy(torch.cat(z_mean_encoded_2motion_list)).reshape((num_test_1motion*2*step, -1), order="A"))

    # ---- plot data ----
    vis = MotionVectorPlot(title="")
    x   = np.linspace(1, step, step)
    z_embedded_motion1 = z_embedded[:num_test_1motion*step].reshape((num_test_1motion, step, -1), order="A")
    z_embedded_motion2 = z_embedded[num_test_1motion*step:].reshape((num_test_1motion, step, -1), order="A")

    cmap       = plt.get_cmap('jet', num_test_1motion*2)
    markersize = np.array([10+(n*20) for n in range(step)])

    # for i in range(num_test_1motion):
    vis.scatter_circle(z_embedded_motion1[0], np.tile(np.array(cmap(0)).reshape(1, -1), (step, 1)), markersize, label="motion{}_domain{}".format(1, "A"))
    vis.scatter_circle(z_embedded_motion1[1], np.tile(np.array(cmap(1)).reshape(1, -1), (step, 1)), markersize, label="motion{}_domain{}".format(1, "B"))
    vis.scatter_cross( z_embedded_motion2[0], np.tile(np.array(cmap(2)).reshape(1, -1), (step, 1)), markersize, label="motion{}_domain{}".format(2, "C"))
    vis.scatter_cross( z_embedded_motion2[1], np.tile(np.array(cmap(3)).reshape(1, -1), (step, 1)), markersize, label="motion{}_domain{}".format(2, "D"))
    vis.save_fig(save_path=os.path.join(test.save_dir, "motion_vector.png"))
