from itertools import repeat
from os import path

import numpy as np
import torch
import torch_geometric.transforms as T
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, InMemoryDataset, download_url


class HyperSpectralCustomDataset(InMemoryDataset):
    """https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html"""

    def __init__(
        self,
        root,
        url,
        gt_url,
        num_neighbours=10,
        train_val_split=(30, 15),
        mat_key=None,
        gt_mat_key=None,
        transform=None,
        pre_transform=None,
    ):
        self.url = url
        self.filename = url.split("/")[-1]
        self.gt_filename = gt_url.split("/")[-1]
        self.gt_url = gt_url
        self.train_val_split = train_val_split
        self.mat_key = mat_key
        self.gt_mat_key = gt_mat_key
        self.num_neighbours = num_neighbours
        self.processed_file = f"{self.mat_key}-k{self.num_neighbours}.pt"
        self.result_path = path.join(root, self.processed_file)
        self.base_transform = T.Compose(
            [
                T.AddTrainValTestMask(
                    "test_rest",
                    num_train_per_class=self.train_val_split[0],
                    num_val=self.train_val_split[1],
                ),
            ]
        )
        super().__init__(root=root, pre_transform=pre_transform, transform=transform)
        self.data, self.slices = torch.load(self.result_path)

    @property
    def processed_file_names(self):
        return [self.processed_file]

    @property
    def raw_file_names(self):
        return [self.filename, self.gt_filename]

    def download(self):
        download_url(self.url, self.raw_dir)
        download_url(self.gt_url, self.raw_dir)

    def process(self):
        data_mat = loadmat(path.join(self.raw_dir, self.filename))
        gt_mat = loadmat(path.join(self.raw_dir, self.gt_filename))
        data = data_mat[self.mat_key]
        gt = gt_mat[self.gt_mat_key]

        pixels = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        pixels_gt = gt.ravel()
        pixels_gt = pixels_gt[pixels_gt != 0]
        filtered_pixels = pixels[pixels_gt]
        pixels_gt -= 1

        neigh = NearestNeighbors(n_neighbors=self.num_neighbours)
        neigh.fit(filtered_pixels)
        neighbours = neigh.kneighbors(filtered_pixels)[1]

        num_of_nodes = filtered_pixels.shape[0]
        num_of_features = filtered_pixels.shape[1]
        x = np.zeros([num_of_nodes, num_of_features]).astype(np.float32)
        y = np.zeros(num_of_nodes).astype(np.int64)
        edge_index = []

        for index, (pixel, gt, neigh) in enumerate(
            zip(filtered_pixels, pixels_gt, neighbours)
        ):
            x[index] = pixel
            y[index] = gt
            edges = list(zip(repeat(index), neigh)) + list(zip(neigh, repeat(index)))
            edge_index.extend(edges)

        x_tensor = torch.as_tensor(x, dtype=torch.float)
        edge_index_tensor = (
            torch.as_tensor(edge_index, dtype=torch.long).t().contiguous()
        )
        y_tensor = torch.as_tensor(y, dtype=torch.long)

        self.data = Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor)

        self.data = self.base_transform(self.data)

        if self.pre_transform is not None:
            self.data = self.pre_transform(self.data)

        self.data, self.slices = self.collate([self.data])
        torch.save((self.data, self.slices), self.result_path)
