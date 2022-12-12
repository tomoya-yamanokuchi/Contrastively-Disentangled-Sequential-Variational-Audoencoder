from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from typing import Optional
from .ActionNormalizedValve import ActionNormalizedValve



class ActionNormalizedValveDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir   = data_dir
        self.batch_size = 32

    def prepare_data(self):
        # download
        print("no implementation for download data")

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full = ActionNormalizedValve(self.data_dir, train=True, transform=self.transform)
            assert full.num_data == 2000
            self.num_train       = 1800
            self.num_val         = 200
            self.train, self.val = random_split(full, [self.num_train, self.num_val])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = ActionNormalizedValve(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.predict = ActionNormalizedValve(self.data_dir, train=False, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)

    @property
    def transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(), # numpy (H x W x C) [0, 255] --> torch (C x H x W) [0.0, 1.0]
                # transforms.Normalize(
                #     mean = (0.6228671625689327, 0.6377793987055758, 0.6407482041398562), # train dataset の [0,1] 変換後の mean
                #     std  = (0.4035978685941395, 0.3913997249338733, 0.39240566963941836) # train dataset の [0,1] 変換後の std
                # ),
            ]
        )


if __name__ == '__main__':
    mnist = ActionNormalizedValveDataModule("./")
