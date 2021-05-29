from typing import Sequence

from datamodules.hyperspectral_datamodule import HyperSpectralDataModule


class PaviaUniversityDataModule(HyperSpectralDataModule):
    def __init__(
            self,
            url: str = "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
            gt_url: str = "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
            data_dir: str = "data/pavia_university",
            num_neighbours: int = 10,
            mat_key: str = "paviaU",
            gt_mat_key: str = "paviaU_gt",
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
        return 103

    @property
    def num_edge_features(self):
        return 0

    @property
    def num_nodes(self):
        return 42776

    @property
    def num_classes(self):
        return 9
