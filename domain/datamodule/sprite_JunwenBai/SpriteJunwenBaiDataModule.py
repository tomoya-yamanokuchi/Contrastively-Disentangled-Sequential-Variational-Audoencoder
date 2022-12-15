import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from typing import Optional

from .SpriteJunwenBai_FastLoad import SpriteJunwenBai_FastLoad as SpriteJunwenBai
# from .SpriteJunwenBai_SlowLoad import SpriteJunwenBai_SlowLoad as SpriteJunwenBai

import torch
torch.multiprocessing.set_start_method('spawn')


class SpriteJunwenBaiDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int=128):
        super().__init__()
        self.data_dir      = data_dir
        self.batch_size    = batch_size
        self.num_dataset   = 11664
        self.num_train     = 9000
        self.num_valid     = 2664
        assert (self.num_train + self.num_valid) == self.num_dataset


    def prepare_data(self):
        # download
        print("no implementation for download data")


    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            data = pickle.load(open(self.data_dir + '/train.pkl', 'rb'))
            self.train = SpriteJunwenBai(data_dir=self.data_dir, train=True)

            data = pickle.load(open(self.data_dir + '/test.pkl', 'rb'))
            self.val   = SpriteJunwenBai(data_dir=self.data_dir, train=False)
            assert self.train.__len__ + self.test.__len__ == self.num_dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = SpriteJunwenBai(data_dir=self.data_dir, train=False)

        if stage == "predict" or stage is None:
            self.predict = SpriteJunwenBai(data_dir=self.data_dir, train=False)


    def train_dataloader(self):
        return DataLoader(
            dataset     = self.train,
            batch_size  = self.batch_size,
            shuffle     = True,
            drop_last   = True,
            pin_memory  = False,
            num_workers = 0,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset     = self.val,
            batch_size  = self.batch_size,
            shuffle     = True,
            drop_last   = True,
            pin_memory  = False,
            num_workers = 0,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset     = self.test,
            batch_size  = 128,
            shuffle     = False,
            drop_last   = True,
            pin_memory  = False,
            num_workers = 0,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset     = self.predict,
            batch_size  = 128,
            shuffle     = False,
            drop_last   = True,
            pin_memory  = False,
            num_workers = 0,
        )

