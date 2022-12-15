# -*- coding: utf-8 -*-
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from domain.test.TestModel import TestModel
from domain.classifier.ClassifierSprite import ClassifierSprite as Classifier
from custom.utility.image_converter import torch2numpy
import cv2; cv2.namedWindow('img', cv2.WINDOW_NORMAL)


log = "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221210035007]-[remote_3090]-32219"
log = "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221212212525]-[remote_3090]-momo"
log = "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221212235346]-[dl-box]-nene"
log = "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221212231238]-[melco]-neko"
log = "[c-dsvae]-[sprite_jb]-[dim_f=256]-[dim_z=32]-[100epoch]-[20221212212403]-[melco]-neko"


def test_cls(dataloader, classifier):
    num_slice  = 1
    _step      = 0
    for index, img_dict in dataloader:
        img = img_dict["images"]
        for test_index in range(len(img)):
            print("[{}-{}] - [{}/{}]".format(index.min(), index.max(), test_index+1, len(img)))
            num_batch, step = img.shape[:2]


if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../conf", config_name="config_classifier")
    def test_cls(cfg: DictConfig) -> None:
        model   = "cdsvae4"
        log_dir = "/hdd_mount/logs_cdsvae/{}/".format(model)
        test    = TestModel(
            config_dir  = log_dir + log,
            checkpoints = "last.ckpt"
        )
        model      = test.load_model()
        dataloader = test.load_dataloader()
        # -------------------------------------
        # import ipdb; ipdb.set_trace()
        classifier         = Classifier(cfg.model.network)
        # cfg.resume         = './judges/Sprite/sprite_judge.tar'
        loaded_dict        = torch.load('./judges/Sprite/sprite_judge.tar')
        classifier.load_state_dict(loaded_dict['state_dict'])
        classifier         = classifier.cuda().eval()


    test_cls()
