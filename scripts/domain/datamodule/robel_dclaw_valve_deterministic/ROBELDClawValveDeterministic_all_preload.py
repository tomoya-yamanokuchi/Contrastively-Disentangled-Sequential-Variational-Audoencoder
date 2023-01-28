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


def get_uminmax(mode):
    '''
    データセット全体から予め計算しておく必要あり
    '''
    u_minmax = {
        'task_space_diff_position': (-0.050, 0.050),
        'joint_space_position'    : (-0.43635426000464594, 0.3996889727516365),
    }
    return u_minmax[mode]


class ROBELDClawValveDeterministic_all_preload(VisionDataset):
    ''' Sprite Dataset
        - sequence
            - train: 2400
            - test : 300
        - step              : 25
        - image size        : (3, 64, 64)
        - action variation  : 6 (初期値でmotionがわかるようになってる)
        - minmax value:
            - min: -1.0
            - max:  1.0
    '''

    def __init__(self, data_dir: str, train: bool, data_type: dict):
        self.data_dir              = data_dir
        self.train                 = train
        self.data_type             = data_type
        self.img_paths             = self._get_img_paths()
        self.num_data              = len(self.img_paths)
        self.min                   = 0
        self.max                   = 255
        self.u_min, self.u_max     = get_uminmax(self.data_type["ctrl_type"])
        self.content_augumentation = ContentAugumentation()
        self.motion_augumentation  = MotionAugumentation(min=self.min, max=self.max)
        self.__all_preload()


    def _get_img_paths(self):
        """
        指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        if self.train: img_dir = self.data_dir + "/dataset_dclaw_deterministic_train2400/"
        else         : img_dir = self.data_dir + "/dataset_dclaw_deterministic_test300/"
        img_dir = Path(img_dir)
        img_paths = [p for p in img_dir.iterdir() if p.suffix == ".db"]
        img_paths = natsorted(img_paths)
        # import ipdb; ipdb.set_trace()
        return img_paths


    def __all_preload(self):
        self.images = []
        self.c_aug  = []
        self.m_aug  = []
        self.ctrl   = []
        for path in self.img_paths:
            path_without_suffix = str(path.resolve()).split(".")[0]          #; print(path_without_suffix)
            db                  = shelve.open(path_without_suffix, flag='r') # read only
            img_numpy           = db["image"]["canonical"]                   # 複数ステップ分が含まれている(1系列分)

            # import ipdb; ipdb.set_trace()
            ctrl                = db["ctrl"][self.data_type["ctrl_type"]]

            step, width, height, channel = img_numpy.shape                   # channlの順番に注意（保存形式に依存する）
            assert channel == 3

            assert img_numpy.min() >= self.min
            assert img_numpy.max() <= self.max
            assert      ctrl.min() >= self.u_min, print("ctrl.min() >= self.u_min = [{}, {}]".format(ctrl.min(), self.u_min))
            assert      ctrl.max() <= self.u_max, print("ctrl.max() >= self.u_max = [{}, {}]".format(ctrl.max(), self.u_max))

            img_torch = self.to_tensor_image(img_numpy)
            img_torch = normalize(x=img_torch, x_min=self.min, x_max=self.max, m=0, M=1)
            self.images.append(img_torch)
            self.c_aug.append(self.content_augumentation.augment(img_torch))
            self.m_aug.append(self.motion_augumentation.augment(img_torch))

            ctrl_tensor = self.to_tensor_ctrl(ctrl)
            ctrl_tensor = normalize(x=ctrl_tensor, x_min=self.u_min, x_max=self.u_max, m=0, M=1)
            self.ctrl.append(ctrl_tensor)
            # print(ctrl_tensor)
            # self.ctrl.append(ctrl)

        self.images = torch.stack(self.images, axis=0)
        self.c_aug  = torch.stack(self.c_aug, axis=0)
        self.m_aug  = torch.stack(self.m_aug, axis=0)
        self.ctrl   = torch.stack(self.ctrl, axis=0)
        # import ipdb; ipdb.set_trace()


    def __len__(self):
        """ディレクトリ内の画像ファイルの数を返す。
        """
        return len(self.img_paths)


    def __getitem__(self, index: int):
        return index, {
            "images" : self.images[index].cuda(),
            "c_aug"  : self.c_aug[index].cuda(),
            "m_aug"  : self.m_aug[index].cuda(),
            # "ctrl"   : self.ctrl[index].cuda(),
            "index"  : index,
        }

    def to_tensor_image(self, pic):
        assert len(pic.shape) == 4 # (step, width, height, channel)
        return torch.from_numpy(pic).permute((0, 3, 1, 2)).contiguous()

    def to_tensor_ctrl(self, ctrl):
        assert len(ctrl.shape) == 2 # (step, dim_u)
        return torch.from_numpy(ctrl).contiguous().type(torch.FloatTensor)

    def to_tensor_label(self, label):
        # assert len(pic.shape) == 4 # (step, width, height, channel)
        return torch.from_numpy(label).contiguous()