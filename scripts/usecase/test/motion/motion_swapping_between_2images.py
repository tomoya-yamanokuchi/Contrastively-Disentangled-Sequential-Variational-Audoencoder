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
from custom import to_numpy, save_image
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
        config_datamodule    = OmegaConf.load("./conf/datamodule/action_norm_valve.yaml")
        datamodule = DataModuleFactory().create(**config_datamodule)
        datamodule.setup(stage="test")
        self.test_dataloader = datamodule.test_dataloader()


    def evaluate(self):
        use_index    = [0, 7] # compare two images
        num_use_test = len(use_index)
        for index, data in self.test_dataloader:
            x = data['images'] [use_index]
            b = data['sensors'][use_index]
            y = data['state']  [use_index]
            u = data['ctrl']   [use_index]
            # << encode original data >>
            return_dict      = self.model(x, b, u)
            z_sample_encoded = return_dict["z_sample"]
            z_sample_prior   = return_dict["z_sample_prior"]
            x_recon          = return_dict["x_recon"]

            # import ipdb; ipdb.set_trace()
            perm            = [1, 0] #torch.randperm(num_use_test); print("perm = ", perm)
            x_gen          , b_gen           = self.model.decode(z=z_sample_prior,       f=return_dict["f_sample"])
            x_gen_randperm , b_gen_randperm  = self.model.decode(z=z_sample_prior[perm], f=return_dict["f_sample"])
            x_swap_randperm, b_swap_randperm = self.model.decode(z=z_sample_encoded[perm], f=return_dict["f_sample"])
            break

        save_image(save_path=os.path.join(test.save_dir, "x.png"),                  image=x,                normalize=True)
        save_image(save_path=os.path.join(test.save_dir, "x_recon.png"),            image=x_recon,          normalize=True)
        save_image(save_path=os.path.join(test.save_dir, "x_gen.png"),              image=x_gen,            normalize=True)
        save_image(save_path=os.path.join(test.save_dir, "x_gen_randperm.png"),     image=x_gen_randperm,   normalize=True)
        save_image(save_path=os.path.join(test.save_dir, "x_swap_randperm.png"),    image=x_swap_randperm,  normalize=True)
        import ipdb; ipdb.set_trace()
        return z_mean_encoded, z_mean_prior



if __name__ == '__main__':

    group_model  = "cdsvae_dclaw_ctrl_5"

    model_cdsvae = "[c-dsvae]-[action_norm_valve]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230119151208]-[remote_tsukumo3090ti]-popo"
    model_cdsvae = "[c-dsvae]-[action_norm_valve]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230123041622]-[remote_tsukumo3090ti]-conv_content"
    model_cdsvae = "[c-dsvae]-[action_norm_valve]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230126024438]-[remote_tsukumo3090ti]-kkk"
    model_cdsvae = "[c-dsvae]-[action_norm_valve]-[dim_f=8]-[dim_z=12]-[300epoch]-[20230126090633]-[remote_tsukumo3090ti]-vvv"


    test = TestDClaw()
    test.load_model(group=group_model, model=model_cdsvae)
    test.load_evaluation_dataset()

    z_mean_encoded, z_mean_prior = test.evaluate()

    z_mean_encoded = z_mean_encoded[:8]
    z_mean_prior   = z_mean_prior  [:8]

    num_test, step, dim_z = z_mean_encoded.shape
    x = np.linspace(1, step, step)

    z_mean_encoded_2d = z_mean_encoded.reshape((num_test*step, dim_z), order="A")
    z_mean_prior_2d   = z_mean_prior.reshape((num_test*step, dim_z), order="A")

    # ---- t-SNE learning ----
    tsne       = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=25, n_iter=1000)
    z_embedded = tsne.fit_transform(np.concatenate((z_mean_encoded_2d, z_mean_prior_2d), axis=0))

    # ---- plot data ----
    vis = MotionVectorPlot(title="")
    z_embedded_encoded = z_embedded[:num_test*step].reshape((num_test, step, -1), order="A")
    z_embedded_prior   = z_embedded[num_test*step:].reshape((num_test, step, -1), order="A")

    cmap       = plt.get_cmap('jet', num_test)
    markersize = np.array([10+(n*20) for n in range(step)])

    for i in range(num_test):
        color = np.tile(np.array(cmap(i)).reshape(1, -1), (step, 1))
        vis.scatter_circle(z_embedded_encoded[i], color, markersize, label="seq{}".format(i))
        vis.scatter_cross(   z_embedded_prior[i], color, markersize, label="seq{}".format(i))
    vis.save_fig(save_path=os.path.join(test.save_dir, "motion_vector.png"))