import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from typing import Optional
from omegaconf import DictConfig
from .SpriteJunwenBai_with_myaug import SpriteJunwenBai_with_myaug as SpriteJunwenBai


class SpriteJunwenBaiDataModule(pl.LightningDataModule):
    def __init__(self, config_dataloader: DictConfig, data_dir: str = "./", **kwargs):
        super().__init__()
        self.sub_name          = kwargs["sub_name"]
        self.config_dataloader = config_dataloader
        self.data_dir          = data_dir
        self.num_dataset       = 11664
        self.num_train         = 9000
        self.num_valid         = 2664
        assert (self.num_train + self.num_valid) == self.num_dataset


    def prepare_data(self):
        print("no implementation for download data")


    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = SpriteJunwenBai(data_dir=self.data_dir, train=True)
            self.val   = SpriteJunwenBai(data_dir=self.data_dir, train=False)
            assert self.val.__len__() == self.num_valid, "{} != {}".format(self.val.__len__, self.num_valid)
            assert self.train.__len__() + self.val.__len__() == self.num_dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = SpriteJunwenBai(data_dir=self.data_dir, train=False)

        if stage == "predict" or stage is None:
            self.predict = SpriteJunwenBai(data_dir=self.data_dir, train=False)


    def train_dataloader(self):
        return DataLoader(self.train, **self.config_dataloader.train)

    def val_dataloader(self):
        return DataLoader(self.val, **self.config_dataloader.except_train)

    def test_dataloader(self):
        return DataLoader(self.test, **self.config_dataloader.except_train)

    def predict_dataloader(self):
        return DataLoader(self.predict, **self.config_dataloader.except_train)

