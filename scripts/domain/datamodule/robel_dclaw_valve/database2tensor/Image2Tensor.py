import numpy as np
import torch
from custom import assert_range, normalize, to_tensor



class Image2Tensor:
    def __init__(self):
        self.min = 0
        self.max = 255


    def get(self, db):
        img_numpy = db["image"]["canonical"]            # 複数ステップ分が含まれている(1系列分)
        assert_range(img_numpy, self.min, self.max)
        step, width, height, channel = img_numpy.shape  # channlの順番に注意（保存形式に依存する）
        assert channel == 3
        img_torch = self.to_tensor_image(img_numpy)     # channlの順番に注意（保存形式に依存する)
        img_torch = normalize(x=img_torch, x_min=self.min, x_max=self.max, m=0, M=1)
        return img_torch


    def to_tensor_image(self, pic):
        assert len(pic.shape) == 4 # (step, width, height, channel)
        return torch.from_numpy(pic).permute((0, 3, 1, 2)).contiguous()

