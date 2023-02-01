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
        self.x = []
        self.x_recon = []


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
            f_encoded_mean = to_numpy(return_dict["f_mean"])
            x_recon        = to_numpy(return_dict["x_recon"])
            x_numpy        = to_numpy(x)

            self.x             .append(x_numpy)
            self.x_recon       .append(x_recon)
            self.index         .append(index)
            self.f_encoded_mean.append(f_encoded_mean)
            self.motion_label  .append(to_numpy(motion_label))

        self.x              = np.concatenate(test.x)
        self.x_recon        = np.concatenate(test.x_recon)
        self.index          = np.concatenate(test.index)
        self.f_encoded_mean = np.concatenate(test.f_encoded_mean)
        self.content_label  = np.array(test.content_class_list)
        self.motion_label   = np.concatenate(test.motion_label)


    def extract_1content_class(self, query_content_label: int):
        index_content_query = list(np.where(self.content_label==query_content_label)[0])
        num_content_i       = len(index_content_query)
        assert (num_content_i == 9) or (num_content_i == 8)
        x              = self.x             [index_content_query]
        x_recon        = self.x_recon       [index_content_query]
        f_encoded_mean = self.f_encoded_mean[index_content_query]
        content_label  = self.content_label [index_content_query]
        motion_label   = self.motion_label  [index_content_query]

        save_image(save_path=os.path.join(self.save_dir, "x_conten{}.png"       .format(query_content_label)), image=torch.Tensor(x      ), normalize=True)
        save_image(save_path=os.path.join(self.save_dir, "x_recon_content{}.png".format(query_content_label)), image=torch.Tensor(x_recon), normalize=True)
        return x, f_encoded_mean, content_label, motion_label


    def extract_n_content_class(self, num_extract_content_label: int, num_action: int):
        x=[]; f_encoded_mean=[]; content_label=[]; motion_label=[]
        for i in range(num_extract_content_label):
            if len(x) >= num_extract_content_label: break
            _x, _f_encoded_mean, _content_label, _motion_label = self.extract_1content_class(i)
            if _x.shape[0] == num_action:
                x.append(_x); f_encoded_mean.append(_f_encoded_mean); content_label.append(_content_label); motion_label.append(_motion_label)
        x              = np.concatenate(x)
        f_encoded_mean = np.concatenate(f_encoded_mean)
        content_label  = np.concatenate(content_label)
        motion_label   = np.concatenate(motion_label)
        return x, f_encoded_mean, content_label, motion_label


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
    model_cdsvae = "[c-dsvae_high_dim]-[sprite_JunwenBai]-[dim_f=256]-[dim_z=32]-[100epoch]-[20230201073453]-[remote_tsukumo3090ti]-high_dim"

    group_model  = "cdsvae_sprite_revisit"

    test = TestDClaw()
    test.load_model(group=group_model, model=model_cdsvae)
    test.load_evaluation_dataset()
    test.evaluate()

    num_action = 9
    x, f_encoded_mean, content_label, motion_label = test.extract_n_content_class(num_extract_content_label=20, num_action=num_action)

    # ---- t-SNE learning ----
    tsne       = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=8, n_iter=1000)
    f_embedded = tsne.fit_transform(f_encoded_mean)

    # ---- plot data ----
    vis         = MotionVectorPlot(title="")

    # ---- plot data ----

    # cmap       = plt.get_cmap('jet', max(content_label))
    # markersize = [10+(n*50) for n in range(num_action)]

    # num_data = f_encoded_mean.shape[0]
    # for i in range(num_data):
    #     print("num = {}/{}".format(i+1, num_data))
    #     index_motion = int(motion_label[i])
    #     content_num  = content_label[i]
    #     vis.scatter_circle(f_embedded[i].reshape(1, -1), np.array(cmap(content_num)).reshape(1, -1), markersize[index_motion], label="content{}".format(content_num))
    #     vis.write_text(f_embedded[i].reshape(1, -1), str(index_motion), fontsize=9)
    # vis.save_fig(save_path=os.path.join(test.save_dir, "content_2D.png"), dpi=500)

    # print("max(content_label) = ", max(content_label))

    # ---- plot data ----
    vis         = MotionVectorPlot(title="")
    num_content = max(content_label)
    text        = ["{}".format(m) for m in range(num_action)]

    cmap       = plt.get_cmap('jet', num_content)
    markersize = np.array([10+(n*50) for n in range(num_action)])
    import ipdb; ipdb.set_trace()
    for i, f_embedded_1content in enumerate(np.split(f_embedded, num_content, axis=0)):
        # import ipdb; ipdb.set_trace()
        vis.scatter_circle(f_embedded_1content, np.tile(np.array(cmap(i)).reshape(1, -1), (num_action, 1)), markersize, label="content{}".format(i))
        vis.write_text(f_embedded_1content, text, fontsize=9)
    vis.save_fig(save_path=os.path.join(test.save_dir, "content_2D.png"), dpi=500)
