import os
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

# ------------------------------------
import sys; import pathlib; p=pathlib.Path("./"); sys.path.append(str(p.parent.resolve()))
from domain.model.ModelFactory import ModelFactory
from domain.classifier.ClassifierJunwenBi.classifier_Sprite_all import classifier_Sprite_all
from domain.classifier import utils
from domain.test.TestModel import TestModel
from domain.test import metric
# ------------------------------------
from custom.utility import reoder


@hydra.main(version_base=None, config_path="../../conf", config_name="config_classifier")
def main(cfg: DictConfig):
    opt = cfg.model

    opt.model = '[c-dsvae]-[sprite_JunwenBi]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221221093615]-[remote_3090]-my_aug'
    opt.model = '[c-dsvae]-[sprite_JunwenBi]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221221093617]-[melco]-my_aug'

    opt.group = 'cdsvae_datamodule_sprite_JunwenBi'


    # ----------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    if opt.model != '':
        log_dir = os.path.join(opt.log_dir, opt.group, opt.model)
        test    = TestModel(
            config_dir  = log_dir,
            checkpoints = "last.ckpt"
        )
        cdsvae      = test.load_model()
        test_loader = test.load_dataloader()
    else:
        raise ValueError('missing checkpoint')

    log = os.path.join(log_dir, 'log.txt')
    os.makedirs('%s/gen/' % log_dir, exist_ok=True)
    os.makedirs('%s/plots/' % log_dir, exist_ok=True)
    dtype = torch.cuda.FloatTensor


    # --------- training loop ------------------------------------
    for epoch in range(opt.niter):

        # print("Epoch", epoch)
        cdsvae.eval()
        mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
        mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        pred1_all, pred2_all, label2_all = list(), list(), list()
        label_gt = list()
        for i, data in test_loader:
            x = data['images']
            recon_x_sample, recon_x = cdsvae.forward_fixed_motion_for_classification(x)


            with torch.no_grad():
                '''
                pred_action1 : originalのテストデータからのactionの予測
                pred_action2 : fixed_motion & varying content のテストデータからのactionの予測
                '''
                pred_action1, pred_skin1, pred_pant1, pred_top1, pred_hair1 = classifier(x)
                pred_action2, pred_skin2, pred_pant2, pred_top2, pred_hair2 = classifier(recon_x_sample)
                pred_action3, pred_skin3, pred_pant3, pred_top3, pred_hair3 = classifier(recon_x)

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


        # import ipdb; ipdb.set_trace()
        label2_all = np.hstack(label2_all)  # label2_all = List[(num_batch,),(num_batch,),...]
        label_gt   = np.hstack(label_gt)
        pred1_all  = np.vstack(pred1_all)
        pred2_all  = np.vstack(pred2_all)

        # import ipdb; ipdb.set_trace()
        acc             = (label_gt == label2_all).mean()
        kl              = metric.KL_divergence(pred2_all, pred1_all)

        nSample_per_cls = min([(label_gt==i).sum() for i in np.unique(label_gt)])
        index           = np.hstack([np.nonzero(label_gt == i)[0][:nSample_per_cls] for i in np.unique(label_gt)]).squeeze()
        pred2_selected  = pred2_all[index]

        IS              = metric.inception_score(pred2_selected)
        H_yx            = metric.entropy_Hyx(pred2_selected)
        H_y             = metric.entropy_Hy(pred2_selected)

        # print('acc: {:.2f}%, kl: {:.4f}, IS: {:.4f}, H_yx: {:.4f}, H_y: {:.4f}'.format(acc*100, kl, IS, H_yx, H_y))
        print('Epoch[{}/{}] : [acc[%], IS, H_yx, H_y] = [{:.2f}, {:.4f}, {:.4f}, {:.4f}]'.format(
            epoch, opt.niter, acc*100, IS, H_yx, H_y))


if __name__ == '__main__':
    main()
