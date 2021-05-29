from typing import Optional, Sequence

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric import transforms as T
from torch_geometric.data import DataLoader, Dataset

from src.datamodules.datasets.sp_fashion_mnist_dataset import FashionMNISTSuperpixelsDataset


class FashionMNISTSuperpixelsDataModule(LightningDataModule):
    """DataModule which converts FashionMNIST image dataset to superpixel graphs.
    Conversion happens on first run only.
    When changing pre_transforms you need to manually delete previously generated dataset files!

    --------Example--------

        from datamodules.mnist_datamodule import FashionMNISTSuperpixelsDataModule
        from pytorch_lightning import seed_everything

        # seeding enables split reproducibility
        seed_everything(123)

        dm = FashionMNISTSuperpixelsDataModule()
        dm.prepare_data()
        dm.setup()

        for batch in dm.train_dataloader():
            x, y, pos, edge_index, batch = batch.x, batch.y, batch.pos, batch.edge_index, batch.batch

    -----------------------

    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Sequence[int] = (55_000, 5_000, 10_000),
        n_segments: int = 100,
        max_num_neighbors: int = 8,
        r: int = 10,
        loop: bool = True,
        **kwargs,
    ):
        """
        Args:
            data_dir: Path to data folder.
            batch_size: Batch size.
            num_workers: Number of processes for data loading.
            pin_memory: Whether to pin CUDA memory (slight speed up for GPU users).
            train_val_test_split: Number of datapoints for training, validation and testing. Should sum up to 70_000.
            n_segments: Desired number of superpixels per image.
            max_num_neighbors: Maximum number of edges for each node/superpixel.
            r: Connect node/superpixel to all nodes in range r (based on superpixel position).
            loop: Whether to add self-loops to nodes.
            **kwargs: Extra parameters passed to SLIC algorithm, learn more here:
                        https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
        """
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_test_split = train_val_test_split
        self.n_segments = n_segments
        self.max_num_neighbors = max_num_neighbors
        self.r = r
        self.loop = loop
        self.slic_kwargs = kwargs

        assert 1 <= n_segments <= 28 * 28

        self.pre_transform = T.Compose(
            [
                T.NormalizeScale(),
            ]
        )
        self.transform = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_node_features(self) -> int:
        return 1

    @property
    def num_edge_features(self) -> int:
        return 0

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download data if needed. Generate superpixel graphs. Apply pre-transforms."""
        FashionMNISTSuperpixelsDataset(
            root=self.data_dir,
            n_segments=self.n_segments,
            max_num_neighbors=self.max_num_neighbors,
            r=self.r,
            loop=self.loop,
            pre_transform=self.pre_transform,
            **self.slic_kwargs,
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset = FashionMNISTSuperpixelsDataset(
            root=self.data_dir,
            n_segments=self.n_segments,
            max_num_neighbors=self.max_num_neighbors,
            r=self.r,
            loop=self.loop,
            pre_transform=self.pre_transform,
            transform=self.transform,
            **self.slic_kwargs,
        )
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, self.train_val_test_split
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
