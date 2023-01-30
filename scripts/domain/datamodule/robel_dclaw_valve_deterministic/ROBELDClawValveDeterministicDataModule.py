import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, random_split
from typing import Optional
from omegaconf import DictConfig
from .ROBELDClawValveDeterministic_all_preload import ROBELDClawValveDeterministic_all_preload as ROBELDClawValveDeterministic


class ROBELDClawValveDeterministicDataModule(pl.LightningDataModule):
    def __init__(self, config_dataloader: DictConfig, data_dir: str = "./", **kwargs):
        super().__init__()
        self.sub_name          = kwargs["sub_name"]
        self.data_type         = kwargs["data_type"]
        self.config_dataloader = config_dataloader
        self.data_dir          = data_dir
        self.num_dataset       = 2400
        self.num_train         = 2200
        self.num_valid         = 200
        assert (self.num_train + self.num_valid) == self.num_dataset


    def prepare_data(self):
        # download
        print("no implementation for download data")


    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full = ROBELDClawValveDeterministic(self.data_dir, train=True, data_type=self.data_type)
            assert full.num_data == self.num_dataset
            # self.train, self.val = random_split(full, [self.num_train, self.num_valid])

            subset_train_indices = list(range(0, self.num_train))                # Ex.) [0,1,.....47999]
            subset_valid_indices = list(range(self.num_train, self.num_dataset)) # Ex.) [48000,48001,.....59999]
            self.train = Subset(full, subset_train_indices)
            self.val   = Subset(full, subset_valid_indices)


        if stage == "test" or stage is None:
            self.test = ROBELDClawValveDeterministic(self.data_dir, train=False, data_type=self.data_type)

        if stage == "predict" or stage is None:
            self.predict = ROBELDClawValveDeterministic(data_dir=self.data_dir, train=False, data_type=self.data_type)


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

