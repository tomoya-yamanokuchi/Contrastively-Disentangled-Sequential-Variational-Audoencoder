import os
import sys
import cv2
import torch
import numpy as np
from pprint import pprint
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import shelve
from natsort import natsorted
from pathlib import Path
from typing import List, Tuple, Optional, Callable
# import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from ..image_augumentation.ContentAugumentation import ContentAugumentation
from ..image_augumentation.MotionAugumentation import MotionAugumentation
from custom.utility.normalize import normalize


class ActionNormalizedValve_all_preload(VisionDataset):
    ''' Sprite Dataset
        - sequence
            - train: 2000
            - test : 200
        - step              : 25
        - image size        : (3, 64, 64)
        - action variation  : 8
            - claw1だけ動く
            - claw2だけ動く
            - claw3だけ動く
            - ３本同時に動く（左右）など
        - minmax value:
            - min: -1.0
            - max:  1.0
    '''

    def __init__(self, data_dir: str, train: bool):
        self.data_dir              = data_dir
        self.train                 = train
        self.img_paths             = self._get_img_paths()
        self.num_data              = len(self.img_paths)
        self.min                   = 0
        self.max                   = 255
        self.content_augumentation = ContentAugumentation()
        self.motion_augumentation  = MotionAugumentation(min=self.min, max=self.max)
        self.__all_preload()


    def _get_img_paths(self):
        """
        指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        if self.train: img_dir = self.data_dir + "/dataset_202210221514_valve2000_train/"
        else         : img_dir = self.data_dir + "/dataset_20221022153117_valve200_test/"
        img_dir = Path(img_dir)
        img_paths = [p for p in img_dir.iterdir() if p.suffix == ".db"]
        img_paths = natsorted(img_paths)
        # import ipdb; ipdb.set_trace()
        return img_paths


    def __all_preload(self):
        self.images = []
        self.c_aug  = []
        self.m_aug  = []
        for path in self.img_paths:
            path_without_suffix = str(path.resolve()).split(".")[0]          #; print(path_without_suffix)
            db                  = shelve.open(path_without_suffix, flag='r') # read only
            img_numpy           = db["image"]["canonical"]                   # 複数ステップ分が含まれている(1系列分)

            # step, width, height, channel = img_numpy.shape                   # channlの順番に注意（保存形式に依存する）
            # assert channel == 3

            img_torch           = self.to_tensor_image(img_numpy)
            img_torch           = normalize(x=img_torch, x_min=self.min, x_max=self.max, m=0, M=1)
            self.images.append(img_torch)
            self.c_aug.append(self.content_augumentation.augment(img_torch))
            self.m_aug.append(self.motion_augumentation.augment(img_torch))
        self.images = torch.stack(self.images, axis=0)
        self.c_aug  = torch.stack(self.c_aug, axis=0)
        self.m_aug  = torch.stack(self.m_aug, axis=0)



    def __len__(self):
        """ディレクトリ内の画像ファイルの数を返す。
        """
        return len(self.img_paths)


    def __getitem__(self, index: int):
        return index, {
            "images" : self.images[index].cuda(),
            "c_aug"  : self.c_aug[index].cuda(),
            "m_aug"  : self.m_aug[index].cuda(),
            "index"  : index,
        }

    def to_tensor_image(self, pic):
        assert len(pic.shape) == 4 # (step, width, height, channel)
        return torch.from_numpy(pic).permute((0, 3, 1, 2)).contiguous()

    def to_tensor_label(self, label):
        # assert len(pic.shape) == 4 # (step, width, height, channel)
        return torch.from_numpy(label).contiguous()