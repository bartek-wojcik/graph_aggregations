import os
from typing import Callable, Optional
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import RadiusGraph, ToSLIC
from torchvision import transforms as T
from torchvision.datasets import VOCDetection


class VocSuperpixelsDataset(InMemoryDataset):
    """Dataset which converts VOC to superpixel graphs (on first run only)."""

    def __init__(
            self,
            root: str = "data/",
            n_segments: int = 500,
            max_num_neighbors: int = 8,
            r: float = 10,
            loop: bool = True,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            **kwargs,
    ):
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                           'train', 'tvmonitor']
        self.data_dir = root
        self.n_segments = n_segments
        self.max_num_neighbors = max_num_neighbors
        self.r = r
        self.loop = loop
        self.slic_kwargs = kwargs
        self.base_transform = T.Compose(
            [
                T.ToTensor(),
                ToSLIC(n_segments=n_segments, **kwargs),
                RadiusGraph(r=r, max_num_neighbors=max_num_neighbors, loop=loop),
            ]
        )
        super().__init__(os.path.join(root, "VOC"), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        """Dynamically generates filename for processed dataset based on superpixel parameters."""
        filename = ""
        filename += f"sp({self.n_segments})_"
        filename += f"maxn({self.max_num_neighbors})_"
        filename += f"r({self.r})_"
        filename += f"loop({self.loop})"
        for name, value in self.slic_kwargs.items():
            filename += f"_{name}({value})"
        filename += ".pt"
        return filename

    def encode_labels(self, label):
        objects = filter(lambda o: o['difficult'] == 0, label['annotation']['object'])
        labels = map(lambda o: o['name'], objects)
        return [category in labels for category in self.categories]

    def download(self):
        VOCDetection(
            self.data_dir, year='2012', image_set='trainval', download=True, transform=self.base_transform
        )

    def process(self):
        dataset = VOCDetection(
            self.data_dir, year='2012', image_set='trainval', download=True, transform=self.base_transform
        )

        # convert to superpixels
        data_list = []
        for graph, label in tqdm(
                dataset, desc="Generating superpixels", colour="GREEN"
        ):
            datapoint = graph
            datapoint.y = torch.tensor(self.encode_labels(label))
            data_list.append(datapoint)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
