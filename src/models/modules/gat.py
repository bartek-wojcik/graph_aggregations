from torch import nn
from torch_geometric.nn import (
    GATConv,
    global_max_pool,
)
import torch.nn.functional as F


class GAT(nn.Module):

    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams

        if hparams["num_conv_layers"] < 1:
            raise Exception("Invalid number of layers!")

        self.conv_modules = nn.ModuleList()

        heads = hparams.get("heads", 1)

        self.conv_modules.append(
            GATConv(hparams["num_node_features"], hparams["conv_size"], heads=heads)
        )

        for _ in range(hparams["num_conv_layers"] - 1):
            conv = GATConv(heads * hparams["conv_size"], hparams["conv_size"], heads=heads)
            self.conv_modules.append(conv)

        self.lin = nn.Linear(hparams["conv_size"], hparams["lin_size"])

        self.output = nn.Linear(hparams["lin_size"], hparams["output_size"])

    def forward(self, x, edge_index, batch, pos):
        for index, layer in enumerate(self.conv_modules):
            x = layer(x, edge_index)
            if index != self.hparams["num_conv_layers"]:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        x = global_max_pool(x, batch)

        x = F.relu(self.lin(x))

        return self.output(x)
