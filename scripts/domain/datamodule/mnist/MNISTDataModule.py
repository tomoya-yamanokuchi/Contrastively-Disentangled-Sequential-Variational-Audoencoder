import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from typing import Optional
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
            mnist_full = MNIST(self.data_dir, train=False, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [9000, 1000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=128)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=5000)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=128)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=128)

    @property
    def transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(64)),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)), # conver gray-scale to RGB
                transforms.Normalize((0.1307,), (0.3081,)),           # conver gray-scale to RGB
            ]
        )


if __name__ == '__main__':
    mnist = MNISTDataModule("./")
    print("MNISTDataModule")
