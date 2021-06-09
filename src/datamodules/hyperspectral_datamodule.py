from typing import Callable, Optional, Sequence

from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader, Dataset, RandomNodeSampler

from src.datamodules.datasets.hyperspectral_dataset import HyperSpectralCustomDataset


class HyperSpectralDataModule(LightningDataModule):
    """
        Base DataModule which converts spectral datasets to graphs.
        Conversion happens on first run only.

        --------Example--------

        from datamodules.hyperspectral_datamodule import HyperSpectralDataModule

        dm = HyperSpectralDataModule(
            url='http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
            gt_url='http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
            data_dir='data/pavia_university',
            mat_key='paviaU',
            gt_mat_key='paviaU_gt',
        )
        dm.prepare_data()
        dm.setup()

        for batch in dm.train_dataloader():
            x, y, edge_index, batch, train_mask, val_mask, test_mask = batch.x, batch.y, batch.edge_index, \
            batch.batch, batch.train_mask, batch.val_mask, batch.test_mask

        -----------------------
    """

    def __init__(
            self,
            url: str,
            gt_url: str,
            data_dir: str = "data/",
            num_neighbours: int = 10,
            mat_key: str = "",
            gt_mat_key: str = "",
            batch_size: int = 1,
            train_val_split: Sequence[int] = (30, 15),
            num_workers: int = 0,
            pin_memory: bool = False,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
    ):
        """
        Args:
            url:                    URL to matlab data file
            gt_url:                 URL to matlab ground truth data file
            data_dir:               Path do data folder
            num_neighbours:         Number of nearest neighbours connected with node
            mat_key:                Matlab dict key where data is stored
            gt_mat_key:             Matlab ground truth key where data is stored
            batch_size:             Batch size (1 - hyperspectral datasets consist of one graph)
            train_val_split:        Number of nodes for training, validation per class (remaining for test)
            num_workers:            Number of processes for data loading.
            pin_memory:             Whether to pin CUDA memory (slight speed up for GPU users)
        """
        super().__init__()
        self.url = url
        self.gt_url = gt_url
        self.data_dir = data_dir
        self.mat_key = mat_key
        self.gt_mat_key = gt_mat_key
        self.num_neighbours = num_neighbours
        self.transform = transform
        self.pre_transform = pre_transform
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        HyperSpectralCustomDataset(
            self.data_dir,
            url=self.url,
            gt_url=self.gt_url,
            mat_key=self.mat_key,
            train_val_split=self.train_val_split,
            gt_mat_key=self.gt_mat_key,
            num_neighbours=self.num_neighbours,
            transform=self.transform,
            pre_transform=self.pre_transform,
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset = HyperSpectralCustomDataset(
            self.data_dir,
            url=self.url,
            gt_url=self.gt_url,
            mat_key=self.mat_key,
            train_val_split=self.train_val_split,
            gt_mat_key=self.gt_mat_key,
            num_neighbours=self.num_neighbours,
            transform=self.transform,
            pre_transform=self.pre_transform,
        )

        self.data_train = dataset
        self.data_val = dataset
        self.data_test = dataset

    def train_dataloader(self):
        return RandomNodeSampler(
            self.data_train.data,
            num_parts=6,
            num_workers=self.num_workers,
            shuffle=True
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
