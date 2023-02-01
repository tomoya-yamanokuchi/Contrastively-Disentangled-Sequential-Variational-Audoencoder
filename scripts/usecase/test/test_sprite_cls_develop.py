
import os
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
# ------------------------------------
import sys; import pathlib; p=pathlib.Path("./"); sys.path.append(str(p.parent.resolve()))
from domain.datamodule.DataModuleFactory import DataModuleFactory
from domain.model.ModelFactory import ModelFactory
from domain.test.TestModel import TestModel
from domain.test import metric_for_gaussian as metric
from domain.visualize.DClawDataPlot import DClawDataPlot
from custom import to_numpy
from custom import reparameterize
from custom import logsumexp
from custom import log_density_z
from custom import save_image

# ------------------------------------
import sys; import pathlib; p=pathlib.Path("./"); sys.path.append(str(p.parent.resolve()))
from domain.classifier.ClassifierJunwenBi.classifier_Sprite_all import classifier_Sprite_all
from domain.test.TestModel import TestModel
from domain.test import metric
# ------------------------------------



class TestDClaw:
    def load_model(self, group, model):
        self.model_name = model
        log_dir         = "/hdd_mount/logs_cdsvae/{}/{}".format(group, model)
        test            = TestModel(config_dir=log_dir, checkpoints="last.ckpt")
        self.model, self.config_model = test.load_model()


    def load_evaluator(self, config_classifier):
        self.config_eval = config_classifier
        classifier       = classifier_Sprite_all(config_classifier)
        loaded_dict      = torch.load(config_classifier.resume)
        classifier.load_state_dict(loaded_dict['state_dict'])
        self.classifier  = classifier.cuda().eval()


    def load_evaluation_dataset(self):
        # << evaluation dataset >>
        datamodule = DataModuleFactory().create(**self.config_model.datamodule)
        datamodule.setup(stage="test")
        self.test_dataloader = datamodule.test_dataloader()


    def evaluate(self):
        acc_list  = []
        IS_list   = []
        H_yx_list = []
        H_y_list  = []

        for epoch in range(self.config_eval.niter):
            mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
            mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
            pred1_all, pred2_all, label2_all = list(), list(), list()
            label_gt = list()
            for i, data in self.test_dataloader:
                x       = data['images']
                label_A = data['A_label']
                label_D = data['D_label']
                c_aug   = data['c_aug']
                m_aug   = data['m_aug']

                if self.config_eval.type_gt == "action": recon_x_sample, recon_x = self.model.forward_fixed_motion_for_classification(x)
                # else:                       recon_x_sample, recon_x = self.model.forward_fixed_content_for_classification(x)

                with torch.no_grad():
                    pred_action1, pred_skin1, pred_pant1, pred_top1, pred_hair1 = self.classifier(x)
                    pred_action2, pred_skin2, pred_pant2, pred_top2, pred_hair2 = self.classifier(recon_x_sample)
                    pred_action3, pred_skin3, pred_pant3, pred_top3, pred_hair3 = self.classifier(recon_x)

                    pred1 = F.softmax(pred_action1, dim = 1)
                    pred2 = F.softmax(pred_action2, dim = 1)
                    pred3 = F.softmax(pred_action3, dim = 1)

                label1 = np.argmax(pred1.detach().cpu().numpy(), axis=1)
                label2 = np.argmax(pred2.detach().cpu().numpy(), axis=1)
                label3 = np.argmax(pred3.detach().cpu().numpy(), axis=1)
                label2_all.append(label2)

                pred1_all.append(pred1.detach().cpu().numpy())
                pred2_all.append(pred2.detach().cpu().numpy())
                label_gt.append(label_D.detach().cpu().numpy())

                def count_D(pred, label, mode=1):
                    return (pred//mode) == (label//mode)

                acc0_sample = count_D(np.argmax(pred_action2.detach().cpu().numpy(), axis=1), label_D.cpu().numpy()).mean()
                acc1_sample = (np.argmax(pred_skin2.detach().cpu().numpy(), axis=1) == label_A[:, 0].cpu().numpy()).mean()
                acc2_sample = (np.argmax(pred_pant2.detach().cpu().numpy(), axis=1) == label_A[:, 1].cpu().numpy()).mean()
                acc3_sample = (np.argmax(pred_top2.detach().cpu().numpy(), axis=1) ==  label_A[:, 2].cpu().numpy()).mean()
                acc4_sample = (np.argmax(pred_hair2.detach().cpu().numpy(), axis=1) == label_A[:, 3].cpu().numpy()).mean()
                mean_acc0_sample += acc0_sample
                mean_acc1_sample += acc1_sample
                mean_acc2_sample += acc2_sample
                mean_acc3_sample += acc3_sample
                mean_acc4_sample += acc4_sample

            label2_all = np.hstack(label2_all)  # label2_all = List[(num_batch,),(num_batch,),...]
            label_gt   = np.hstack(label_gt)
            pred1_all  = np.vstack(pred1_all)
            pred2_all  = np.vstack(pred2_all)

            acc = (label_gt == label2_all).mean()
            # kl  = metric.KL_divergence(pred2_all, pred1_all)

            nSample_per_cls = min([(label_gt==i).sum() for i in np.unique(label_gt)])
            index           = np.hstack([np.nonzero(label_gt == i)[0][:nSample_per_cls] for i in np.unique(label_gt)]).squeeze()
            pred2_selected  = pred2_all[index]

            IS   = metric.inception_score(pred2_selected)
            H_yx = metric.entropy_Hyx(pred2_selected)
            H_y  = metric.entropy_Hy(pred2_selected)

            print('Epoch[{}/{}] : [acc[%], IS, H_yx, H_y] = [{:.2f}, {:.4f}, {:.4f}, {:.4f}]'.format(epoch, self.config_eval.niter, acc*100, IS, H_yx, H_y))
            acc_list.append(acc); IS_list.append(IS); H_yx_list.append(H_yx); H_y_list.append(H_y)

        acc_list_mean  = np.mean(acc_list)
        IS_list_mean   = np.mean(IS_list)
        H_yx_list_mean = np.mean(H_yx_list)
        H_y_list_mean  = np.mean(H_y_list)

        print("--------------------------------------------------------------------------")
        print('model mean : [acc[%], IS, H_yx, H_y] = [{:.2f}, {:.4f}, {:.4f}, {:.4f}] : model={}'.format(
            acc_list_mean*100, IS_list_mean, H_yx_list_mean, H_y_list_mean, self.model_name))
        print("--------------------------------------------------------------------------")
        return acc_list_mean, IS_list_mean, H_yx_list_mean, H_y_list_mean




if __name__ == '__main__':

    # search_key = "[remote_3090]-revisit"
    # search_key = "[remote_tsukumo3090ti]-revisit"
    search_key = "[remote_tsukumo3090ti]-high_dim"

    group_model       = "cdsvae_sprite_revisit"
    pathlib_obj       = pathlib.Path("/hdd_mount/logs_cdsvae/{}".format(group_model))
    model_cdsvae_list = [str(model).split("/")[-1] for model in list(pathlib_obj.glob("*")) if search_key in str(model)]

    num_eval_per_model = 1


    for m, model_cdsvae in enumerate(model_cdsvae_list):
        config_classifier = OmegaConf.load("./conf/model/classifier_sprite.yaml")

        test = TestDClaw()
        test.load_model(group=group_model, model=model_cdsvae)
        test.load_evaluator(config_classifier)
        test.load_evaluation_dataset()

        test.evaluate()
