from torch import nn
from torch_geometric.nn import (
    global_max_pool,
)
import torch.nn.functional as F
from src.models.modules.vector_sage import VectorSAGE


class VectorSAGEModule(nn.Module):

    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams

        if hparams["num_conv_layers"] < 1:
            raise Exception("Invalid number of layers!")

        self.conv_modules = nn.ModuleList()

        self.conv_modules.append(
            VectorSAGE(hparams["num_node_features"], hparams["conv_size"], spatial_dim=2, aggregator='mean')
        )

        for _ in range(hparams["num_conv_layers"] - 1):
            conv = VectorSAGE(hparams["conv_size"], hparams["conv_size"], spatial_dim=2, aggregator='mean')
            self.conv_modules.append(conv)

        self.lin = nn.Linear(hparams["conv_size"], hparams["lin_size"])

        self.output = nn.Linear(hparams["lin_size"], hparams["output_size"])

    def forward(self, x, edge_index, batch, pos):
        for index, layer in enumerate(self.conv_modules):
            x = layer(x, edge_index, pos)
            if index != self.hparams["num_conv_layers"]:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        x = global_max_pool(x, batch)

        x = F.relu(self.lin(x))

        return self.output(x)
