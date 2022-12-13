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
        data                             = pickle.load(open(self.data_dir + '/data.pkl', 'rb'))
        X_train, X_test, A_train, A_test = data['X_train'], data['X_test'], data['A_train'], data['A_test']
        D_train, D_test                  = data['D_train'], data['D_test']
        c_augs_train, c_augs_test        = data['c_augs_train'], data['c_augs_test']
        m_augs_train, m_augs_test        = data['m_augs_train'], data['m_augs_test']

        train_data = SpriteJunwenBai(train=True, data = X_train, A_label = A_train,
                            D_label = D_train, c_aug = c_augs_train, m_aug = m_augs_train)
        test_data  = SpriteJunwenBai(train=False, data = X_test, A_label = A_test,
                            D_label = D_test, c_aug = c_augs_test, m_aug = m_augs_test)

        assert len(train_data) + len(test_data) == self.num_dataset

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = train_data
            self.val   = test_data

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = test_data

        if stage == "predict" or stage is None:
            self.predict = test_data


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
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=9, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, shuffle=False)

