from typing import Sequence

from src.datamodules.hyperspectral_datamodule import HyperSpectralDataModule


class KennedySpaceCenterDataModule(HyperSpectralDataModule):
    def __init__(
            self,
            url: str = "http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat",
            gt_url: str = "http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat",
            data_dir: str = "data/kennedy_space_center",
            num_neighbours: int = 10,
            mat_key: str = "KSC",
            gt_mat_key: str = "KSC_gt",
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
        return 176

    @property
    def num_edge_features(self):
        return 0

    @property
    def num_nodes(self):
        return 5211

    @property
    def num_classes(self):
        return 13
