import torch
from torch import nn
from torch_geometric.nn import (
    SAGEConv,
    global_max_pool,
)


class GraphSage(nn.Module):

    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams

        if hparams["num_conv_layers"] < 1:
            raise Exception("Invalid number of layers!")

        activation = nn.ReLU

        if hparams["aggregation_method"] == "concat":
            self.aggregation_method = None
        elif hparams["aggregation_method"] == "mean":
            self.aggregation_method = None
        elif hparams["aggregation_method"] == "max":
            self.aggregation_method = None
        else:
            raise Exception("Invalid aggregation method name")

        self.conv_modules = nn.ModuleList()
        self.activ_modules = nn.ModuleList()

        self.conv_modules.append(
            SAGEConv(hparams["num_node_features"], hparams["conv_size"])
        )
        self.activ_modules.append(activation())

        for _ in range(hparams["num_conv_layers"] - 1):
            conv = SAGEConv(hparams["conv_size"], hparams["conv_size"])
            self.conv_modules.append(conv)
            self.activ_modules.append(activation())

        self.lin = nn.Linear(hparams["conv_size"], hparams["lin_size"])

        self.output = nn.Linear(hparams["lin_size"], hparams["output_size"])

    def forward(self, data):
        x, edge_index, batch, pos = data.x, data.edge_index, data.batch, data.pos
        x = torch.cat((x, pos), 1)
        for layer, activation in zip(self.conv_modules, self.activ_modules):
            x = layer(x, edge_index)
            x = activation(x)

        x = global_max_pool(x, batch)

        x = self.lin(x)

        return self.output(x)
