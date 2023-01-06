import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from typing import Optional
from omegaconf import DictConfig

# from .ActionNormalizedValve import ActionNormalizedValve
from .ActionNormalizedValve_all_preload import ActionNormalizedValve_all_preload as ActionNormalizedValve


class ActionNormalizedValveDataModule(pl.LightningDataModule):
    def __init__(self, config_dataloader: DictConfig, data_dir: str = "./", **kwargs):
        super().__init__()
        self.sub_name          = kwargs["sub_name"]
        self.ctrl_type         = kwargs["ctrl_type"]
        self.config_dataloader = config_dataloader
        self.data_dir          = data_dir
        self.num_dataset       = 2000
        self.num_train         = 1800
        self.num_valid         = 200
        assert (self.num_train + self.num_valid) == self.num_dataset


    def prepare_data(self):
        # download
        print("no implementation for download data")


    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full = ActionNormalizedValve(self.data_dir, train=True, ctrl_type=self.ctrl_type)
            assert full.num_data == self.num_dataset
            self.train, self.val = random_split(full, [self.num_train, self.num_valid])

        if stage == "test" or stage is None:
            self.test = ActionNormalizedValve(data_dir=self.data_dir, train=False, ctrl_type=self.ctrl_type)

        if stage == "predict" or stage is None:
            self.predict = ActionNormalizedValve(data_dir=self.data_dir, train=False, ctrl_type=self.ctrl_type)


    def train_dataloader(self):
        return DataLoader(self.train, **self.config_dataloader.train)

    def val_dataloader(self):
        return DataLoader(self.val, **self.config_dataloader.val)

    def test_dataloader(self):
        return DataLoader(self.test, **self.config_dataloader.test)

    def predict_dataloader(self):
        return DataLoader(self.predict, **self.config_dataloader.predict)

    def grouped_dataloader(self):
        return DataLoader(self.test, **self.config_dataloader.grouped)

