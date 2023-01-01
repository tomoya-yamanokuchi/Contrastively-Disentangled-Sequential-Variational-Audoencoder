from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from typing import Optional
from omegaconf import DictConfig
from .SpriteAugmentation import SpriteAugmentation


class SpriteAugmentationDataModule(pl.LightningDataModule):
    def __init__(self, config_dataloader: DictConfig, data_dir: str = "./"):
        super().__init__()
        self.config_dataloader = config_dataloader
        self.data_dir          = data_dir
        self.full_num_data     = 6687
        self.num_train         = 5350
        self.num_valid         = 1337
        assert (self.num_train + self.num_valid) == self.full_num_data


    def prepare_data(self):
        print("\n\n No implementation for download data \n\n")


    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full = SpriteAugmentation(self.data_dir, train=True)
            assert full.num_data == self.full_num_data
            self.train, self.val = random_split(full, [self.num_train, self.num_valid])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = SpriteAugmentation(self.data_dir, train=False)

        if stage == "predict" or stage is None:
            self.predict = SpriteAugmentation(self.data_dir, train=False)


    def train_dataloader(self):
        return DataLoader(self.train, **self.config_dataloader.train)
        # return DataLoader(self.val, **self.config_dataloader.except_train)

    def val_dataloader(self):
        return DataLoader(self.val, **self.config_dataloader.except_train)

    def test_dataloader(self):
        return DataLoader(self.test, **self.config_dataloader.except_train)

    def predict_dataloader(self):
        return DataLoader(self.predict, **self.config_dataloader.except_train)


