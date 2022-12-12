import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset



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
        self.train     = train
        self.transform = transform
        self.img_paths = self._get_img_paths(img_dir)
        self.num_data  = len(self.img_paths)
        self.min       =  -1.0
        self.max       =   1.0

        self.transform_color_dist   = self.get_transform_color_distortion()
        self.transform_GaussianBlur = self.get_transform_GaussianBlur()

    def _get_img_paths(self, img_dir):
        """
        指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        if self.train: img_dir = img_dir + "/Sprite/lpc-dataset/train/"
        else         : img_dir = img_dir + "/Sprite/lpc-dataset/test"
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

        img_aug_context  = self.augment_context(img)
        img_aug_dynamics = self.augment_dynamics(img)
        return index, (img, img_aug_context, img_aug_dynamics)


    def augment_context(self, img):
        '''
        - keep  : context
        - change: dynamics
        '''
        step         = img.shape[0]
        random_index = torch.randperm(step)
        return torch.index_select(img, dim=0, index=random_index)


    def normalize(self, x, x_min, x_max, m, M):
        a = (x - x_min) / (x_max - x_min)
        return a * (M - m) + m


    def augment_dynamics(self, img):
        '''
        - keep  : dynamics
        - change: context

        combination of
            - (cropping)
            - color distortion
            - Gaussian blur
            - reshaping
        '''
        img = self.normalize(img, x_min=self.min, x_max=self.max, m=0, M=1) # [-1, 1] to [0, 1]
        # -----------------------------------------------------------------
        img = self.transform_color_dist(img)
        img = torch.cat([self.transform_GaussianBlur(_img) for _img in torch.split(img, 1, 0)], dim=0)
        # -----------------------------------------------------------------
        img = self.normalize(img, x_min=0, x_max=1, m=self.min, M=self.max) # [0, 1] to [-1, 1]
        return img


    def get_transform_color_distortion(self, s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(
            brightness = 0.8*s,
            contrast   = 0.8*s,
            saturation = 0.8*s,
            hue        = 0.2*s,
        )
        rnd_color_jitter = transforms.RandomApply(transforms=[color_jitter], p=0.8)
        rnd_gray         = transforms.RandomGrayscale(p=0.2)
        return transforms.Compose([rnd_color_jitter, rnd_gray])


    def get_transform_GaussianBlur(self, kernel_size=(3, 3), sigma=(0.1, 2.0)):
        '''
        - kernel_size: set to be 10% of the image height/width
            (if img_size = (64, 64) -> kernel_size=(7, 7) with odd restriction).
        - sigma      : randomly sampled from [0.1, 2.0]
          (Please refere [Ting Chen, et al., ICML2020])
        '''
        assert type(kernel_size) == tuple
        assert type(sigma) == tuple
        blurrer     = transforms.GaussianBlur(kernel_size, sigma)
        rnd_blurrer = transforms.RandomApply(transforms=[blurrer], p=0.5)
        return transforms.Compose([rnd_blurrer])


if __name__ == '__main__':
    import numpy as np
    import cv2
    from torchvision import transforms

    loader  = SpriteAugmentation("data/Sprite/", train=False, transform=transforms.ToTensor())

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    for i,dataitem in enumerate(loader):
        # _,_,_,_,_,_,data = dataitem
        print("loader : ", i)

        images = []
        for k, d in enumerate(dataitem):
            # print("dataitem : ", k)
            d = np.array(d) # (step, channel, w, h)
            dt = np.transpose(d, (1, 2, 0))
            cv2.imshow("img", dt)
            cv2.waitKey(50)

