import cv2
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import torch
import os
import random
import pickle
import numpy as np
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from ..image_augumentation.ContentAugumentation import ContentAugumentation
from ..image_augumentation.MotionAugumentation import MotionAugumentation


class SpriteJunwenBai_with_myaug(VisionDataset):
    ''' Sprite Dataset (shared by Junwen Bai)
        - sequence
            - train: 9000
            - test : 2664
        - step              : 8
        - image size        : (3, 64, 64)
        - motion variation  : 9
            - actions (walking, spellcasting, slashing) x directions (left, front, right)
        - minmax value:
            - min:  0.0
            - max:  1.0
    '''

    def __init__(self, data_dir: str, train: bool):
        self.data_dir = data_dir
        self.train    = train
        self._load_data()
        self.min                   =  0.0
        self.max                   =  1.0
        self.content_augumentation = ContentAugumentation()
        self.motion_augumentation  = MotionAugumentation(min=self.min, max=self.max)


    def _load_data(self):
        if   self.train: name = "train"
        else           : name = "test"
        data = pickle.load(open(self.data_dir + '/{}.pkl'.format(name), 'rb'))
        self.data    = data['X_{}'.format(name)]
        self.A_label = data['A_{}'.format(name)]
        self.D_label = data['D_{}'.format(name)]
        self.c_aug   = data['c_augs_{}'.format(name)]
        self.m_aug   = data['m_augs_{}'.format(name)]
        self.N       = self.data.shape[0]
        self.aug_num = self.c_aug.shape[1]


    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)


    def __len__(self):
        return self.N


    def __getitem__(self, index):
        data_ancher    = self.data[index]    # (8, 64, 64, 3)
        A_label_ancher = self.A_label[index] # (4,)
        D_label_ancher = self.D_label[index] # ()

        data_ancher    = self.to_tensor_image(data_ancher)

        return index, {
            "images" : data_ancher.cuda(),
            "c_aug"  : self.content_augumentation.augment(data_ancher).cuda(),
            "m_aug"  : self.motion_augumentation.augment(data_ancher).cuda(),
            "A_label": self.to_tensor_label(A_label_ancher).cuda(),
            "D_label": self.to_tensor_label(np.array([D_label_ancher])).cuda(),
            "index"  : index,
        }


    def to_tensor_image(self, pic):
        assert len(pic.shape) == 4 # (step, width, height, channel)
        return torch.from_numpy(pic).permute((0, 3, 1, 2)).contiguous()

    def to_tensor_label(self, label):
        # assert len(pic.shape) == 4 # (step, width, height, channel)
        return torch.from_numpy(label).contiguous()