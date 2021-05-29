from typing import Optional, Sequence, Union

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, ConcatDataset
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.transforms import Polar


class MNISTSuperpixelsCustom(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            num_nodes: Union[int, str] = 75,
            batch_size: int = 32,
            train_val_test_split: Sequence[int] = (55_000, 5_000, 10_000),
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if self.num_nodes == "75" or self.num_nodes == 75:
            self.data_dir += "/MNIST_sp_75"
        elif self.num_nodes == "150" or self.num_nodes == 150:
            self.data_dir += "/MNIST_sp_150"
        elif self.num_nodes == "300" or self.num_nodes == 300:
            self.data_dir += "/MNIST_sp_300"
        else:
            raise Exception("Invalid number of nodes.")

        self.transform = None
        self.pre_transform = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        trainset = MNISTSuperpixels(
            self.data_dir, train=True, transform=self.pre_transform, pre_transform=Polar()
        )
        testset = MNISTSuperpixels(
            self.data_dir, train=False, transform=self.pre_transform, pre_transform=Polar()
        )
        dataset_full = ConcatDataset(datasets=[trainset, testset])

        self.data_train, self.data_val, self.data_test = random_split(
            dataset_full, self.train_val_test_split
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
