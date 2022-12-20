import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset

from custom import normalize
import sys; import pathlib; p=pathlib.Path(); sys.path.append(str(p.parent.resolve()))
from ..image_augumentation.ContentAugumentation import ContentAugumentation
from ..image_augumentation.MotionAugumentation import MotionAugumentation



class SpriteAugmentation(VisionDataset):
    ''' Sprite Dataset
        - sequence
            - train: 6687
            - test : 873
        - step              : 8
        - image size        : (3, 64, 64)
        - action variation  : 9
            - 歩いたり，手を振ったりなど
        - minmax value:
            - min: -1.0
            - max:  1.0
    '''

    def __init__(self,
            img_dir  : str,
            train    : bool,
            transform: Optional[Callable] = None,
        ):
        self.train                 = train
        self.transform             = transform
        self.img_paths             = self._get_img_paths(img_dir)
        self.num_data              = len(self.img_paths)
        self.min                   = -1.0
        self.max                   =  1.0
        self.content_augumentation = ContentAugumentation()
        self.motion_augumentation  = MotionAugumentation(min=self.min, max=self.max)


    def _get_img_paths(self, img_dir):
        """
        指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        if self.train: img_dir = img_dir + "/lpc-dataset/train/"
        else         : img_dir = img_dir + "/lpc-dataset/test"
        img_dir = Path(img_dir)
        img_paths = [p for p in img_dir.iterdir() if p.suffix == ".sprite"]
        # import ipdb; ipdb.set_trace()
        img_paths = self._paths_sorted(img_paths)
        return img_paths


    def _paths_sorted(self, paths):
        '''
        ・x.stem はそのままの意味でxが持つstemという変数にアクセスしている
        ・ここでxはpathsの要素になるのでPosixPathに対応する
        ・従って，やっていることは PosixPath(~).stem と等価
        ・PosixPath(~).stem の中には整数値 int が string の形で格納されている
        '''
        return sorted(paths, key = lambda x: int(x.stem))


    def __len__(self):
        """ディレクトリ内の画像ファイルの数を返す。
        """
        return len(self.img_paths)


    def __getitem__(self, index: int):
        path = self.img_paths[index] # Ex.) PosixPath('data/Sprite/lpc-dataset/train/1808.sprite')
        img  = torch.load(str(path)) # img.shape = torch.Size([8, 3, 64, 64]) になるので１つのパスからロードしたデータに複数ステップ分が含まれている
        # step, channel, width, height = img.shape
        # print("min: {} max: {}".format(img.min(), img.max())) # --> min=0.0, max=1.0
        img = normalize(x=img, x_min=self.min, x_max=self.max, m=0, M=1)
        # print("min: {} max: {}".format(img.min(), img.max())) # --> min=0.0, max=1.0
        
        # import ipdb; ipdb.set_trace()
        return index, {
            "images" : img.cuda(),
            "c_aug"  : self.content_augumentation.augment(img).cuda(),
            "m_aug"  : self.motion_augumentation.augment(img).cuda(),
            "index"  : index,
        }
        
    transforms.ToTensor

    def to_tensor_image(self, pic):
        assert len(pic.shape) == 4 # (step, width, height, channel)
        return torch.from_numpy(pic).permute((0, 3, 1, 2)).contiguous()