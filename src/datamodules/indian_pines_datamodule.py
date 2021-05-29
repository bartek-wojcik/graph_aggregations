from typing import Sequence

from src.datamodules.hyperspectral_datamodule import HyperSpectralDataModule


class IndianPinesDataModule(HyperSpectralDataModule):
    def __init__(
            self,
            url: str = "http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat",
            gt_url: str = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
            data_dir: str = "data/indian_pines",
            num_neighbours: int = 10,
            mat_key: str = "indian_pines",
            gt_mat_key: str = "indian_pines_gt",
            batch_size: int = 1,
            train_val_split: Sequence[int] = (30, 15),
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__(
            url,
            gt_url,
            data_dir,
            num_neighbours,
            mat_key,
            gt_mat_key,
            batch_size,
            train_val_split,
            num_workers,
            pin_memory,
        )

    @property
    def num_node_features(self):
        return 220

    @property
    def num_edge_features(self):
        return 0

    @property
    def num_nodes(self):
        return 10249

    @property
    def num_classes(self):
        return 16
