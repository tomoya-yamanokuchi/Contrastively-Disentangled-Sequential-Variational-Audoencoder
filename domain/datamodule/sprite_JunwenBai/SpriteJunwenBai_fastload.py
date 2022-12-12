import cv2
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import torch
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Callable


class SpriteJunwenBai(VisionDataset):
    ''' Sprite Dataset (shared by Junwen Bai)
        - sequence
            - train: 9000
            - test : 2664
        - step              : 8
        - image size        : (3, 64, 64)
        - action variation  : 9
            - 歩いたり，手を振ったりなど
        - minmax value:
            - min:  0.0
            - max:  1.0
    '''

    def __init__(self, train, data, A_label, D_label, c_aug, m_aug):
        self.data    = data
        self.A_label = A_label
        self.D_label = D_label
        self.N       = self.data.shape[0]
        self.c_aug   = c_aug
        self.m_aug   = m_aug
        self.aug_num = c_aug.shape[1]


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
        idx            = np.random.randint(self.aug_num)
        c_aug_anchor   = self.c_aug[index][idx] # (8, 64, 64, 3)
        m_aug_anchor   = self.m_aug[index][idx] # (8, 64, 64, 3)

        # print("min: {} max: {}".format(data_ancher.min(), data_ancher.max())) # --> min=0.0, max=1.0
        # import ipdb; ipdb.set_trace()

        return index, {
            "images" : self.to_tensor_image(data_ancher).cuda(),
            "c_aug"  : self.to_tensor_image(c_aug_anchor).cuda(),
            "m_aug"  : self.to_tensor_image(m_aug_anchor).cuda(),
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