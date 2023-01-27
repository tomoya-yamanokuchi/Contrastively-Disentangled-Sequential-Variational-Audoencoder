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
        # ----- training dataset for calculating marginal predictive distribution ------
        config_datamodule    = OmegaConf.load("./conf/datamodule/sprite_JunwenBai.yaml")

        config_datamodule.config_dataloader.train.shuffle = False

        datamodule = DataModuleFactory().create(**config_datamodule)
        # datamodule.setup(stage="test")
        # self.test_dataloader = datamodule.test_dataloader()
        datamodule.setup(stage="fit")
        self.test_dataloader = datamodule.train_dataloader()



    def evaluate(self):
        for index, data in self.test_dataloader:
            x = data['images']
            f_sample_list = []

            num_content = int(len(index)/8)
            for i in range(num_content):
                # import ipdb; ipdb.set_trace()
                x_inter_domain = x[(i*8):((i+1)*8)].contiguous()

                return_dict      = self.model(x_inter_domain)
                x_recon          = return_dict["x_recon"]
                f_sample         = return_dict["f_sample"]

                save_image(save_path=os.path.join(test.save_dir, "motion{}_x.png"      .format(i)), image=x_inter_domain,      normalize=True)
                f_sample_list.append(f_sample)
            break

        return f_sample_list



if __name__ == '__main__':

    group_model  = "cdsvae_sprite_2"

    model_cdsvae = "[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230127014016]-remote_3090_ddd"
    model_cdsvae = "[c-dsvae]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230127014103]-tsukumo3090ti_ddd"

    test = TestDClaw()
    test.load_model(group=group_model, model=model_cdsvae)
    test.load_evaluation_dataset()

    f_sample_inter_content_list = test.evaluate()
    num_test, dim_z             = f_sample_inter_content_list[0].shape

    # ---- t-SNE learning ----
    tsne       = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=8, n_iter=1000)
    f_embedded = tsne.fit_transform(to_numpy(torch.cat(f_sample_inter_content_list)))

    # ---- plot data ----
    vis         = MotionVectorPlot(title="")
    num_content = len(f_sample_inter_content_list)
    num_action  = 8
    text        = ["{}".format(m) for m in range(num_action)]

    cmap       = plt.get_cmap('jet', num_content)
    markersize = np.array([10+(n*50) for n in range(num_action)])
    for i, f_embedded_1content in enumerate(np.split(f_embedded, num_content, axis=0)):
        # import ipdb; ipdb.set_trace()
        vis.scatter_circle(f_embedded_1content, np.tile(np.array(cmap(i)).reshape(1, -1), (num_action, 1)), markersize, label="domain{}".format(i))
        vis.write_text(f_embedded_1content, text, fontsize=9)
    vis.save_fig(save_path=os.path.join(test.save_dir, "motion_vector.png"), dpi=500)