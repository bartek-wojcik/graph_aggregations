import torch
from torch import nn
from torch_geometric.nn import (
    JumpingKnowledge,
    global_max_pool, GCNConv,
)


class JK(nn.Module):

    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        self.aggregation_method = hparams["aggregation_method"]

        if hparams["num_conv_layers"] < 1:
            raise Exception("Invalid number of layers!")

        activation = nn.ReLU

        self.conv_modules = nn.ModuleList()
        self.activ_modules = nn.ModuleList()

        self.conv_modules.append(
            GCNConv(hparams["num_node_features"], hparams["conv_size"])
        )
        self.activ_modules.append(activation())

        for _ in range(hparams["num_conv_layers"] - 1):
            conv = GCNConv(hparams["conv_size"], hparams["conv_size"])
            self.conv_modules.append(conv)
            self.activ_modules.append(activation())

        if self.aggregation_method == 'lstm':
            self.jk = JumpingKnowledge(self.aggregation_method,
                                       num_layers=hparams["num_conv_layers"],
                                       channels=hparams["conv_size"])
        else:
            self.jk = JumpingKnowledge(self.aggregation_method)

        if self.aggregation_method == 'cat':
            self.lin = nn.Linear(int(hparams["conv_size"] * hparams["num_conv_layers"]), hparams["lin_size"])
        else:
            self.lin = nn.Linear(int(hparams["conv_size"]), hparams["lin_size"])
        self.output = nn.Linear(hparams["lin_size"], hparams["output_size"])

    def forward(self, x, edge_index, batch):
        xs = []
        for layer, activation in zip(self.conv_modules, self.activ_modules):
            x = layer(x, edge_index)
            x = activation(x)
            xs += [x]

        x = self.jk(xs)
        x = global_max_pool(x, batch)

        x = self.lin(x)

        return self.output(x)
