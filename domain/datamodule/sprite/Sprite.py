import cv2
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import torch
import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable


class Sprite(VisionDataset):
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


    def __getitem__(self, index: int):
        path = self.img_paths[index] # Ex.) PosixPath('data/Sprite/lpc-dataset/train/1808.sprite')
        img  = torch.load(str(path)) # img.shape = torch.Size([8, 3, 64, 64]) になるので１つのパスからロードしたデータに複数ステップ分が含まれている
        # step, channel, width, height = img.shape
        # print("min: {} max: {}".format(img.min(), img.max())) # max=1.0, min=-1.0
        # import ipdb; ipdb.set_trace()
        # img_unrolled = img.view(-1, channel, width, height)
        # if self.transform is not None:
            # img = self.transform(img)
        ### img, state, ctrl = load(~)
        ### return img["canonical"], img[""]
        return index, img


    def __len__(self):
        """ディレクトリ内の画像ファイルの数を返す。
        """
        return len(self.img_paths)


if __name__ == '__main__':
    import numpy as np
    import cv2
    from torchvision import transforms

    loader  = Sprite("data/Sprite/", train=False, transform=transforms.ToTensor())

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

