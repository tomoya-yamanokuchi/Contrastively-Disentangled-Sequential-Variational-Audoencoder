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
from domain.test.ContentLabel import ContentLabel
from domain.test.UniqueLabelAssigner import UniqueLabelAssigner
from domain.visualize.save_plot import plot_2D_latent_space
from domain.visualize.MotionVectorPlot import MotionVectorPlot
from custom import to_numpy, save_image, save_image_as_gif
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class TestDClaw:
    def __init__(self):
        self.unique_label_assigner = UniqueLabelAssigner()
        self.content_class_list    = []

        self.index                 = []
        self.f_encoded_mean        = []
        self.content_label         = []
        self.motion_label          = []


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
        '''
            num_action: number of action variation
        '''
        for index, batch in self.test_dataloader:
            x             = batch['images']
            content_label = batch['A_label']
            motion_label  = batch['D_label']
            self.append_content_label(content_label)

            return_dict    = self.model(x)
            self.index.append(index)
            f_encoded_mean = to_numpy(return_dict["f_mean"])
            self.f_encoded_mean.append(f_encoded_mean)
            self.motion_label.append(to_numpy(motion_label))

            # save_image(save_path=os.path.join(test.save_dir, "content{}_x.png".format(i)), image=x,      normalize=True)
            # f_sample_list.append(f_sample)
            # break
        # return f_sample_list


    def append_content_label(self, content_label_batch):
        num_batch = content_label_batch.shape[0]
        for j in range(num_batch):
            clb           = ContentLabel(content_label_batch[j])
            content_class = self.unique_label_assigner.assgin(clb)
            self.content_class_list.append(content_class)


if __name__ == '__main__':


    model_cdsvae = "[c-dsvae]-[sprite_JunwenBai]-[dim_f=8]-[dim_z=12]-[100epoch]-[20230131152725]-[remote_tsukumo3090ti]-revisit"
    model_cdsvae = "[c-dsvae]-[sprite_JunwenBai]-[dim_f=8]-[dim_z=12]-[100epoch]-[20230131161458]-[remote_tsukumo3090ti]-revisit"

    model_cdsvae = "[c-dsvae]-[sprite_JunwenBai]-[dim_f=8]-[dim_z=12]-[100epoch]-[20230131170209]-[remote_tsukumo3090ti]-revisit"

    group_model  = "cdsvae_sprite_revisit"

    test = TestDClaw()
    test.load_model(group=group_model, model=model_cdsvae)
    test.load_evaluation_dataset()
    test.evaluate()

    index          = np.concatenate(test.index)
    f_encoded_mean = np.concatenate(test.f_encoded_mean)
    content_label  = np.array(test.content_class_list)
    motion_label   = np.concatenate(test.motion_label)

    # ---- t-SNE learning ----
    tsne       = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=100, n_iter=10000)
    f_embedded = tsne.fit_transform(f_encoded_mean)

    # ---- plot data ----
    vis         = MotionVectorPlot(title="")

    # ---- plot data ----
    num_action = 9
    cmap       = plt.get_cmap('jet', max(content_label))
    markersize = [10+(n*50) for n in range(num_action)]

    num_data = f_encoded_mean.shape[0]
    for i in range(num_data):
        print("num = {}/{}".format(i+1, num_data))
        index_motion = int(motion_label[i])
        content_num  = content_label[i]
        vis.scatter_circle(f_embedded[i].reshape(1, -1), np.array(cmap(content_num)).reshape(1, -1), markersize[index_motion], label="content{}".format(content_num))
        vis.write_text(f_embedded[i].reshape(1, -1), str(index_motion), fontsize=9)
    vis.save_fig(save_path=os.path.join(test.save_dir, "content_2D.png"), dpi=500)

    print("max(content_label) = ", max(content_label))
