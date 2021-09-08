import os
from typing import Callable, Optional

import torch
from torch.utils.data import ConcatDataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import RadiusGraph, ToSLIC, KNNGraph
from torchvision import transforms as T
from torchvision.datasets import STL10
from tqdm import tqdm


class STL10Dataset(InMemoryDataset):

    def __init__(
            self,
            root: str = "data/",
            n_segments: int = 400,
            k: int = 10,
            loop: bool = True,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            **kwargs,
    ):
        self.data_dir = root
        self.n_segments = n_segments
        self.k = k
        self.loop = loop
        self.slic_kwargs = kwargs
        self.base_transform = T.Compose(
            [
                T.ToTensor(),
                ToSLIC(n_segments=n_segments, add_img=True, compactness=1, **kwargs),
                KNNGraph(k=k, loop=loop),
            ]
        )
        super().__init__(os.path.join(root, "STL10"), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        """Dynamically generates filename for processed dataset based on superpixel parameters."""
        filename = ""
        filename += f"sp({self.n_segments})_"
        filename += f"k({self.k})_"
        filename += f"loop({self.loop})"
        for name, value in self.slic_kwargs.items():
            filename += f"_{name}({value})"
        filename += ".pt"
        return filename

    def download(self):
        STL10(
            self.data_dir, split='train', download=True, transform=self.base_transform
        )
        STL10(
            self.data_dir, split='test', download=True, transform=self.base_transform
        )

    def process(self):
        trainset = STL10(
            self.data_dir, split='train', download=True, transform=self.base_transform
        )
        testset = STL10(
            self.data_dir, split='test', download=True, transform=self.base_transform
        )
        dataset = ConcatDataset(datasets=[trainset, testset])

        # convert to superpixels
        data_list = []
        for graph, label in tqdm(
                dataset, desc="Generating superpixels", colour="GREEN"
        ):
            datapoint = graph
            datapoint.y = torch.tensor(label)
            data_list.append(datapoint)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
